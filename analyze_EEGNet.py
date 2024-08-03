# %%
import os

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import seaborn as sn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold, train_test_split
from sklearn.utils import shuffle
from torch.utils.data import DataLoader, TensorDataset
from torchsummary import summary
from tqdm import tqdm


# %%
class EEGNet(nn.Module):  # EEGNET-8,2
    def __init__(
        self,
        chans=22,
        classes=4,  # Retain the parameter for flexibility, but won't be used in the FC layer
        time_points=257,
        f1=8,
        f2=16,
        d=2,
        dropoutRate=0.5,
        max_norm1=1,
        max_norm2=0.25,
    ):
        super(EEGNet, self).__init__()
        # Calculating FC input features
        linear_input_size = (time_points // 32) * f2

        # Temporal Filters
        self.block1 = nn.Sequential(
            nn.Conv2d(1, f1, (1, 32), padding='same', bias=False),
            nn.BatchNorm2d(f1),
        )
        # Spatial Filters
        self.block2 = nn.Sequential(
            nn.Conv2d(f1, d * f1, (chans, 1), groups=f1, bias=False),  # Depthwise Conv
            nn.BatchNorm2d(d * f1),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropoutRate),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(
                d * f1, f2, (1, 16), groups=f2, bias=False, padding='same'
            ),  # Separable Conv
            nn.Conv2d(f2, f2, kernel_size=1, bias=False),  # Pointwise Conv
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropoutRate),
        )
        self.flatten = nn.Flatten()

        # Change the output of the fully connected layer to 1
        self.fc = nn.Linear(linear_input_size, 1)

        # Apply max_norm constraint to the depthwise layer in block2
        self._apply_max_norm(self.block2[0], max_norm1)

        # Apply max_norm constraint to the linear layer
        self._apply_max_norm(self.fc, max_norm2)

    def _apply_max_norm(self, layer, max_norm):
        for name, param in layer.named_parameters():
            if 'weight' in name:
                param.data = torch.renorm(param.data, p=2, dim=0, maxnorm=max_norm)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


# %%
mne.set_log_level('ERROR')
# Get the data directory
base_dir = os.path.join(os.getcwd(), 'data', 'EEG_preprocessed')
# Get a list of all the subjects
subjects = os.listdir(base_dir)
subjects.pop(167)
subjects.pop(167)

n_epochs_list = []
raw_list = []
epochs_list = []
ch_names = []
shared_ch_names = [
    'P8',
    'P5',
    'CP3',
    'C3',
    'P1',
    'CP5',
    'P6',
    'PO8',
    'PO3',
    'C5',
    'P4',
    'Oz',
    'F4',
    'PO4',
    'P2',
    'C4',
    'C2',
    'Pz',
    'AF3',
]

for i, subj in tqdm(enumerate(subjects), total=len(subjects)):
    # Read in the data
    eo_path = os.path.join(base_dir, subj, f'{subj}_EO.set')
    ec_path = os.path.join(base_dir, subj, f'{subj}_EC.set')
    raw_eo = mne.io.read_raw_eeglab(eo_path).resample(128, npad='auto')
    raw_eo.pick_channels(shared_ch_names)

    epochs = mne.make_fixed_length_epochs(raw_eo, duration=5)
    epochs.drop_bad()
    epochs_list.append(epochs)
    n_epochs_list.append(len(epochs))

all_epochs = mne.concatenate_epochs(epochs_list)

# %%
n_chans = 19
n_times = 640

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = EEGNet(time_points=n_times, chans=n_chans).to(device)
input_size = (1, n_chans, n_times)
summary(model, input_size)

# %%
# Prepare the data
data = all_epochs.get_data()
# Normalizing Input features: z-score(mean=0, std=1)
X = (data - np.mean(data)) / np.std(data)

# Prepare the labels
data_dir = os.path.join(
    os.getcwd(),
    'data',
    'behavioral',
    'Emotion_and_Personality_Test_Battery_LEMON',
)
y = []
df = pd.read_csv(os.path.join(data_dir, 'TAS.csv'), index_col=0).sort_index()
for i, subj in enumerate(subjects):
    y.extend([df['TAS_OverallScore'][subj]] * n_epochs_list[i])

# Checking the existence of null & inf in the dataset
if np.any(np.isnan(X)) or np.any(np.isinf(X)):
    raise ValueError('Data contains NaNs or infinities after normalization.')
if np.any(np.isnan(y)) or np.any(np.isinf(y)):
    raise ValueError('Labels contain NaNs or infinities.')

# Making the X and y tensors for K-Fold Cross Validation
X_tensor = torch.Tensor(X).unsqueeze(1)
y_tensor = torch.LongTensor(y)

# Spliting  Data: 80% for Train and 20% for Test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Converting to Tensor
X_train = torch.Tensor(X_train).unsqueeze(1).to(device)
X_test = torch.Tensor(X_test).unsqueeze(1).to(device)
y_train = torch.LongTensor(y_train).to(device)
y_test = torch.LongTensor(y_test).to(device)

# Creating Tensor Dataset
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

# Printing the sizes
print('Size of X_train:', X_train.size())
print('Size of X_test:', X_test.size())
print('Size of y_train:', y_train.size())
print('Size of y_test:', y_test.size())

# %%
# criterion = nn.MSELoss()
criterion = nn.L1Loss()
learning_rate = 0.01
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

num_epochs = 500
batch_size = 64
loss_list = []
for epoch in range(num_epochs):
    model.train()
    X_train, y_train = shuffle(X_train, y_train)
    running_loss = 0.0
    # correct = 0
    # total = 0
    for i in range(0, len(X_train), batch_size):
        inputs = X_train[i : i + batch_size].to(device)
        labels = y_train[i : i + batch_size].to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        # _, predicted = torch.max(outputs, 1)
        # total += labels.size(0)
        # correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(X_train)
    loss_list.append(epoch_loss)
    # epoch_accuracy = correct / total
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')
average_loss = running_loss / len(X_train)
print('Average Loss:', average_loss)

torch.save(model, 'my_model.pth')

# %%
