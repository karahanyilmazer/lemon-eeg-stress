# %% [markdown]
# https://colab.research.google.com/github/amrzhd/EEGNet/blob/main/EEGNet.ipynb

# %% [markdown]
## Motor Imagery Task Classification

# %% [markdown]
## Extracting Data

# %% [markdown]
### Downloading BCI Competition IV 2a Dataset
# https://www.bbci.de/competition/download/competition_iv/BCICIV_2a_gdf.zip

# %%
import os

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import seaborn as sn
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold, train_test_split
from sklearn.utils import shuffle
from torch.utils.data import DataLoader, TensorDataset
from torchsummary import summary

# %% [markdown]
# # Structuring Data

# %%
raw_data_folder = 'content/raw_data/'
cleaned_data_folder = 'content/cleaned_data/'
files = os.listdir(raw_data_folder)

# Filtering out files with suffix 'E.gdf'
filtered_files = [file for file in files if file.endswith('T.gdf')]

raw_list = []

# Iterating through filtered files
for file in filtered_files:
    file_path = os.path.join(raw_data_folder, file)

    # Reading raw data
    raw = mne.io.read_raw_gdf(
        file_path, eog=['EOG-left', 'EOG-central', 'EOG-right'], preload=True
    )
    # Droping EOG channels
    raw.drop_channels(['EOG-left', 'EOG-central', 'EOG-right'])

    # High Pass Filtering 4-40 Hz
    raw.filter(l_freq=4, h_freq=40, method='iir')

    # Notch filter for Removal of Line Voltage
    raw.notch_filter(freqs=50)

    # Resampling Data
    raw.resample(128, npad='auto')

    # Saving the modified raw data to a file with .fif suffix
    new_file_path = os.path.join(cleaned_data_folder, file[:-4] + '.fif')
    raw.save(new_file_path, overwrite=True)
    # Appending data to the list
    raw_list.append(raw)

final_raw = mne.concatenate_raws(raw_list)
new_file_path = os.path.join(cleaned_data_folder, 'All_Subjects.fif')
final_raw.save(new_file_path, overwrite=True)

# %% [markdown]
# **List of the events**
#
# '1023': 1 Rejected trial
#
# '1072': 2 Eye movements
#
# '276': 3 Idling EEG (eyes open)
#
# '277': 4 Idling EEG (eyes closed)
#
# '32766': 5 Start of a new run
#
# '768': 6 Start of a trial
#
# '769': 7 Cue onset **Left** (class 1) : 0
#
# '770': 8 Cue onset **Right** (class 2) : 1
#
# '771': 9 Cue onset **Foot** (class 3) : 2
#
# '772': 10 Cue onset **Tongue** (class 4): 3

# %%
events = mne.events_from_annotations(final_raw)
events[1]

# %% [markdown]
# **Time choice:**
# [0.5s, 2,5s] Post Cue on set:  [3.75s, 5.75s]

# %%
epochs = mne.Epochs(
    final_raw,
    events[0],
    event_id=[7, 8, 9, 10],
    tmin=0.5,
    tmax=2.5,
    reject=None,
    baseline=None,
    preload=True,
)
data = epochs.get_data(copy=True)
labels = epochs.events[:, -1]

# %%
print("Dataset's shape:", data.shape)

# %%
# Choosing Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Loss Function
criterion = nn.CrossEntropyLoss()

# Normalizing Labels to [0, 1, 2, 3]
y = labels - np.min(labels)

# Normalizing Input features: z-score(mean=0, std=1)
X = (data - np.mean(data)) / np.std(data)

# Checking the existance of null & inf in the dataset
if np.any(np.isnan(X)) or np.any(np.isinf(X)):
    raise ValueError("Data contains NaNs or infinities after normalization.")
if np.any(np.isnan(y)) or np.any(np.isinf(y)):
    raise ValueError("Labels contain NaNs or infinities.")

# Making the X,y tensors for K-Fold Cross Validation
X_tensor = torch.Tensor(X).unsqueeze(1)
y_tensor = torch.LongTensor(y)

# Spliting  Data: 80% for Train and 20% for Test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
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
print("Size of X_train:", X_train.size())
print("Size of X_test:", X_test.size())
print("Size of y_train:", y_train.size())
print("Size of y_test:", y_test.size())


# %% [markdown]
# # EEGNet Model


# %%
class EEGNet(nn.Module):  # EEGNET-8,2
    def __init__(
        self,
        chans=22,
        classes=4,
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
            # nn.BatchNorm2d(f1, momentum=0.01, eps=1e-3),
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
        self.fc = nn.Linear(linear_input_size, classes)

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


# %% [markdown]
# ## Model Summery

# %%
input_size = (1, 22, 257)  # 1716 Trainable Parameters (As mentioned in paper)
eegnet_model = EEGNet().to(device)
summary(eegnet_model, input_size)

# %% [markdown]
# ## Training Loop

# %%
eegnet_model = EEGNet().to(device)
learning_rate = 0.001
optimizer = optim.Adam(eegnet_model.parameters(), lr=learning_rate)

num_epochs = 500
batch_size = 64
for epoch in range(num_epochs):
    eegnet_model.train()
    X_train, y_train = shuffle(X_train, y_train)
    running_loss = 0.0
    correct = 0
    total = 0
    for i in range(0, len(X_train), batch_size):
        inputs = X_train[i : i + batch_size].to(device)
        labels = y_train[i : i + batch_size].to(device)

        optimizer.zero_grad()
        outputs = eegnet_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(X_train)
    epoch_accuracy = correct / total
    print(
        f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {(epoch_accuracy*100):.2f}%"
    )
average_loss = running_loss / len(X_train)
print("Average Loss:", average_loss)

# Saving model
torch.save(eegnet_model, 'eegnet_model.pth')


# %% [markdown]
# ## Testing Model

# %%
eegnet_model.eval()
correct = 0
total = 0
with torch.no_grad():
    for i in range(len(X_test)):
        inputs = X_test[i : i + 1].to(device)
        labels = y_test[i : i + 1].to(device)
        outputs = eegnet_model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = (correct / total) * 100
print(f"Test Accuracy: {accuracy:.2f}%")


# %% [markdown]
# ### Confusion Matrix

# %%
eegnet_model.eval()
y_pred = []
y_true = []
classes = ['Left', 'Right', 'Foot', 'Tongue']

with torch.no_grad():
    for inputs, labels in zip(X_test, y_test):
        outputs = eegnet_model(inputs.unsqueeze(0))  # Forward pass
        _, predicted = torch.max(outputs.data, 1)
        y_pred.append(predicted.item())
        y_true.append(labels.item())

cf_matrix = confusion_matrix(y_true, y_pred)
cf_matrix = cf_matrix.astype('float') / cf_matrix.sum(axis=1)[:, np.newaxis]

# Create DataFrame for visualization
df_cm = pd.DataFrame(cf_matrix, index=classes, columns=classes)

# Plot confusion matrix
plt.figure(figsize=(10, 7))
sn.heatmap(df_cm, annot=True, cmap='Blues', fmt='.2f')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix_eegnet.png')
plt.show()


# %% [markdown]
# ## K-Fold Cross Validation

# %%
k_folds = 4
results = {}
torch.manual_seed(42)
num_epochs = 500
batch_size = 128
learning_rate = 0.001


def reset_weights(m):
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            print(f'Reset trainable parameters of layer = {layer}')
            layer.reset_parameters()


# K-fold Cross Validator
kfold = KFold(n_splits=k_folds, shuffle=True)
fold_accuracies = []

# K-fold Cross Validation model evaluation
for fold, (train_ids, test_ids) in enumerate(
    kfold.split(X), 1
):  # Assuming X contains your data

    print(f"FOLD ({fold})")
    print("--------------------------------")

    # Data loaders for training and testing data in this fold
    X_train_fold, X_test_fold = X[train_ids], X[test_ids]
    y_train_fold, y_test_fold = y[train_ids], y[test_ids]
    # Convert NumPy arrays to PyTorch tensors
    X_train_fold_tensor = torch.Tensor(X_train_fold).unsqueeze(1).to(device)
    X_test_fold_tensor = torch.Tensor(X_test_fold).unsqueeze(1).to(device)
    y_train_fold_tensor = torch.LongTensor(y_train_fold)
    y_test_fold_tensor = torch.LongTensor(y_test_fold)

    # Create Tensor datasets
    train_dataset = TensorDataset(X_train_fold_tensor, y_train_fold_tensor)
    test_dataset = TensorDataset(X_test_fold_tensor, y_test_fold_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    eegnet_model = EEGNet().to(device)
    eegnet_model.apply(
        reset_weights
    )  # Resetting model's weights to avoid weight leakage
    optimizer = optim.Adam(eegnet_model.parameters(), lr=learning_rate)

    # Training Loop
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        eegnet_model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = eegnet_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_accuracy = correct / total
        print(
            f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {(epoch_accuracy*100):.2f}%"
        )

    # Evaluating on test set
    eegnet_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = eegnet_model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_accuracy = correct / total
    print(f"Accuracy for fold {fold}: {(test_accuracy*100):.2f}%")
    print("--------------------------------")
    results[fold] = test_accuracy

print(f"K-Fold Cross Validation Results for {k_folds} Folds")
print("--------------------------------")
avg_accuracy = sum(results.values()) / len(results)
for fold, accuracy in results.items():
    print(f'Fold ({fold}): {accuracy:.2f} %')
print(f'Average Accuracy: {avg_accuracy:.2f} %')


# %% [markdown]
# ### Results of K-Fold Cross Validation

# %%
accuracy_values = list(results.values())
mean_accuracy = np.mean(accuracy_values)
std_accuracy = np.std(accuracy_values)

# Calculate standard error of the mean (SEM)
SEM = std_accuracy / np.sqrt(len(accuracy_values))

# Calculate error bars (2 * SEM)
error = 2 * SEM

# Plot mean accuracy with error bars
plt.errorbar(1, mean_accuracy, yerr=error, fmt='o', capsize=5)
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Mean Accuracy with Error Bars (2 SEM)')
plt.show()

# %%
