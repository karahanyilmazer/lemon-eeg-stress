# %%
import os
import pickle

import mne
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset
from torchsummary import summary
from tqdm import tqdm


# %%
class EEGNet(nn.Module):  # EEGNET-8,2
    def __init__(
        self,
        chans=22,
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


class ModifiedEEGNet(nn.Module):
    def __init__(
        self,
        chans=19,  # Updated channel count
        time_points=640,  # Updated time points
        f1=16,  # Increased initial filter count
        f2=32,  # Increased output filter count for block3
        d=4,  # Increased depth multiplier
        dropoutRate=0.5,
        max_norm1=1,
        max_norm2=0.25,
    ):
        super(ModifiedEEGNet, self).__init__()

        # Adjusted FC input feature calculation based on pool and conv layers
        linear_input_size = f2 * 5  # Correct input size for fc1 after Block 4

        # Temporal Filters
        self.block1 = nn.Sequential(
            nn.Conv2d(
                1, f1, (1, 64), padding='same', bias=False
            ),  # Increased kernel size
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
                d * f1,
                f2,
                (1, 32),
                groups=f2,
                bias=False,
                padding='same',  # Increased kernel size
            ),  # Separable Conv
            nn.Conv2d(f2, f2, kernel_size=1, bias=False),  # Pointwise Conv
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropoutRate),
        )

        # Optional additional convolutional block for increased complexity
        self.block4 = nn.Sequential(
            nn.Conv2d(f2, f2, (1, 16), padding='same', bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropoutRate),
        )

        self.flatten = nn.Flatten()

        # Change the output of the fully connected layer to have more neurons
        self.fc1 = nn.Linear(linear_input_size, 256)  # Correct input feature size
        self.fc2 = nn.Linear(256, 1)

        # Apply max_norm constraint to the depthwise layer in block2
        self._apply_max_norm(self.block2[0], max_norm1)

        # Apply max_norm constraint to the linear layers
        self._apply_max_norm(self.fc1, max_norm2)
        self._apply_max_norm(self.fc2, max_norm2)

    def _apply_max_norm(self, layer, max_norm):
        for name, param in layer.named_parameters():
            if 'weight' in name:
                param.data = torch.renorm(param.data, p=2, dim=0, maxnorm=max_norm)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)  # Optional: include this line if block4 is added
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


# %%
mne.set_log_level('ERROR')
# Get the data directory
base_dir = os.path.join(os.getcwd(), 'data', 'EEG_preprocessed')
# Get a list of all the subjects
bad_subjs = ['sub-032345', 'sub-032357', 'sub-032450', 'sub-032493', 'sub-032513']
subjects = [subj for subj in os.listdir(base_dir) if subj not in bad_subjs]

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
all_chans = [
    'FC4',
    'F7',
    'Oz',
    'TP7',
    'Fz',
    'F4',
    'CPz',
    'CP5',
    'PO4',
    'F6',
    'F8',
    'FC1',
    'P6',
    'F5',
    'TP8',
    'PO8',
    'FT8',
    'FC5',
    'FT7',
    'F3',
    'Fp2',
    'CP2',
    'P3',
    'PO7',
    'T8',
    'P4',
    'O2',
    'PO10',
    'C4',
    'P5',
    'CP4',
    'O1',
    'AF4',
    'PO9',
    'C5',
    'T7',
    'CP3',
    'CP6',
    'Fp1',
    'C6',
    'FC2',
    'Cz',
    'PO3',
    'F1',
    'Pz',
    'AF3',
    'P1',
    'AFz',
    'C2',
    'CP1',
    'P7',
    'AF8',
    'POz',
    'F2',
    'FC3',
    'P8',
    'AF7',
    'C1',
    'P2',
    'C3',
    'FC6',
]

# %%
for i, subj in tqdm(enumerate(subjects), total=len(subjects)):
    # Read in the data
    eo_path = os.path.join(base_dir, subj, f'{subj}_EO.set')
    ec_path = os.path.join(base_dir, subj, f'{subj}_EC.set')
    raw_eo = mne.io.read_raw_eeglab(eo_path, preload=True)
    raw_ec = mne.io.read_raw_eeglab(ec_path, preload=True)
    missing_channels = list(set(all_chans) - set(raw_eo.ch_names))
    for ch in missing_channels:
        raw_eo.add_channels(
            [
                mne.io.RawArray(
                    np.zeros((1, len(raw_eo.times))),
                    mne.create_info([ch], raw_eo.info['sfreq'], ch_types='eeg'),
                )
            ]
        )
    raw_eo.info['bads'] = missing_channels
    raw_eo.set_montage('standard_1020')
    raw_eo = raw_eo.interpolate_bads(reset_bads=True)
    missing_channels = list(set(all_chans) - set(raw_ec.ch_names))
    for ch in missing_channels:
        raw_ec.add_channels(
            [
                mne.io.RawArray(
                    np.zeros((1, len(raw_ec.times))),
                    mne.create_info([ch], raw_ec.info['sfreq'], ch_types='eeg'),
                )
            ]
        )
    raw_ec.info['bads'] = missing_channels
    raw_ec.set_montage('standard_1020')
    raw_eo = raw_eo.interpolate_bads(reset_bads=True)
    raw_eo = raw_eo.resample(128, npad='auto')
    raw_ec = raw_ec.resample(128, npad='auto')
    raw_eo.pick_channels(shared_ch_names)
    raw_ec.pick_channels(shared_ch_names)

    raw = mne.concatenate_raws([raw_eo, raw_ec])

    epochs = mne.make_fixed_length_epochs(raw, duration=5)
    epochs.drop_bad()
    epochs_list.append(epochs)
    n_epochs_list.append(len(epochs))

all_epochs = mne.concatenate_epochs(epochs_list)

# %%
# all_epochs.save('all_epochs-epo.fif.gz')
all_epochs = mne.read_epochs('all_epochs-epo.fif.gz')
# Read the pickle file
with open('n_epochs_list.pkl', 'rb') as pkl_file:
    n_epochs_list = pickle.load(pkl_file)

# %%
n_chans = all_epochs.get_data().shape[1]
n_times = all_epochs.get_data().shape[2]
batch_size = 128

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
# model = EEGNet(time_points=n_times, chans=n_chans).to(device)
model = ModifiedEEGNet(time_points=n_times, chans=n_chans).to(device)
input_size = (1, n_chans, n_times)
summary(model, input_size)

# %%
# Prepare the data
X = all_epochs.get_data()
del all_epochs

# Normalizing Input features: z-score(mean=0, std=1)
X = (X - np.mean(X)) / np.std(X)

# Prepare the labels
data_dir = os.path.join(
    os.getcwd(),
    'data',
    'behavioral',
    'Emotion_and_Personality_Test_Battery_LEMON',
)
y = []
df = pd.read_csv(os.path.join(data_dir, 'PSQ.csv'), index_col=0).sort_index()
for i, subj in enumerate(subjects):
    y.extend([float(df['PSQ_OverallScore'][subj])] * n_epochs_list[i])

# Checking the existence of null & inf in the dataset
if np.any(np.isnan(X)) or np.any(np.isinf(X)):
    raise ValueError('Data contains NaNs or infinities after normalization.')
if np.any(np.isnan(y)) or np.any(np.isinf(y)):
    raise ValueError('Labels contain NaNs or infinities.')

# Spliting  Data: 80% for Train and 20% for Test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=41
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

# Converting to Tensor
X_train = torch.Tensor(X_train).unsqueeze(1).to(device)
X_val = torch.Tensor(X_val).unsqueeze(1).to(device)
X_test = torch.Tensor(X_test).unsqueeze(1).to(device)
y_train = torch.Tensor(y_train).to(device)
y_val = torch.Tensor(y_val).to(device)
y_test = torch.Tensor(y_test).to(device)

# Creating Tensor Dataset
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
test_dataset = TensorDataset(X_test, y_test)

# Create Data Loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Printing the sizes
print('Size of X_train:', X_train.size())
print('Size of X_val:', X_train.size())
print('Size of X_test:', X_test.size())
print('Size of y_train:', y_train.size())
print('Size of y_val:', y_train.size())
print('Size of y_test:', y_test.size())

# %%
criterion = nn.MSELoss()
# criterion = nn.L1Loss()
learning_rate = 0.01
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

# Training loop with early stopping
best_val_loss = float('inf')
patience = 20
trigger_times = 0

# Training Loop with Validation
num_epochs = 300
loss_list = []
val_loss_list = []
r2_list = []

for epoch in range(num_epochs):
    # Training
    # ==================================================================================
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    loss_list.append(epoch_loss)

    # Validation
    # ==================================================================================
    model.eval()
    val_running_loss = 0.0
    all_outputs = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            val_running_loss += loss.item() * inputs.size(0)

            # Collect outputs and labels for R2 score calculation
            all_outputs.append(outputs.squeeze().cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    val_epoch_loss = val_running_loss / len(val_loader.dataset)
    val_loss_list.append(val_epoch_loss)

    # Calculate R2 score
    all_outputs = np.concatenate(all_outputs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    r2 = r2_score(all_labels, all_outputs)
    r2_list.append(r2)

    # Early stopping
    if val_epoch_loss < best_val_loss:
        best_val_loss = val_epoch_loss
        trigger_times = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print('Early stopping!')
            break

    print(
        f'Epoch {epoch+1}/{num_epochs}\t'
        f'Loss: {epoch_loss:.4f}\t'
        f'Val. Loss: {val_epoch_loss:.4f}\t'
        f'R2 Score: {r2:.4f}'
    )

    # Step the scheduler
    scheduler.step()

    if epoch in np.arange(24, 300, 25):
        torch.save(model, f'my_model_{epoch+1}.pth')

average_loss = running_loss / len(train_loader.dataset)
average_val_loss = val_running_loss / len(val_loader.dataset)
print('Average Loss:', average_loss)
print('Average Validation Loss:', average_val_loss)

# %%
model = ModifiedEEGNet().to(device)

# Load the model weights
model_path = 'best_model.pth'
model.load_state_dict(torch.load(model_path, weights_only=False))

# Set the model to evaluation mode
model.eval()

# Lists to store predictions and true labels
all_predictions = []
all_labels = []

with torch.no_grad():  # Disable gradient calculation
    for inputs, labels in test_loader:
        # Forward pass to get predictions
        outputs = model(inputs)

        # Store predictions and true labels
        all_predictions.append(outputs.squeeze().cpu().numpy())
        all_labels.append(labels.cpu().numpy())

# Concatenate all predictions and labels into arrays
all_predictions = np.concatenate(all_predictions, axis=0)
all_labels = np.concatenate(all_labels, axis=0)

# Compute evaluation metrics
mse = mean_squared_error(all_labels, all_predictions)
r2 = r2_score(all_labels, all_predictions)

# Output the results
print(f"Test MSE: {mse:.4f}")
print(f"Test R² Score: {r2:.4f}")

# %%
