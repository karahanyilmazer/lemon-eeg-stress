# %%
#!%matplotlib qt
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'eeg-classes'))
from src.utils.DataLoader import DataLoader  # type: ignore

# %%
# Define the base data directory
data_dir = os.path.join(
    os.getcwd(),
    'data',
    'behavioral',
    'Emotion_and_Personality_Test_Battery_LEMON',
)

# Read the data
df_psq = pd.read_csv(os.path.join(data_dir, 'PSQ.csv'), index_col=0).sort_index()
df_tas = pd.read_csv(os.path.join(data_dir, 'TAS.csv'), index_col=0).sort_index()
df_tic = pd.read_csv(os.path.join(data_dir, 'TICS.csv'), index_col=0).sort_index()

# Get a list of all subjects with EEG data
subj_list = os.listdir(os.path.join(os.getcwd(), 'data', 'EEG_preprocessed'))

# Filter the data based on the subjects
df_psq = df_psq[df_psq.index.isin(subj_list)]
df_tas = df_tas[df_tas.index.isin(subj_list)]
df_tic = df_tic[df_tic.index.isin(subj_list)]

# Load the data
data_loader = DataLoader(os.path.join(os.getcwd(), 'data'))
X_bp_abs = data_loader.load_pkl(os.path.join('feat_mats', 'X_bp_abs'))
X_bp_rel = data_loader.load_pkl(os.path.join('feat_mats', 'X_bp_rel'))

# %%
# Define the feature and label dictionaries
feat_dict = {'X_bp_rel': X_bp_rel, 'X_bp_abs': X_bp_abs}

# Pick the features to use
selected_feat = 'X_bp_rel'
X = feat_dict[selected_feat]

df_y = pd.concat([df_psq, df_tic, df_tas], axis=1)
df_X = pd.DataFrame(
    X,
    columns=[
        'Theta (EO)',
        'Alpha (EO)',
        'Beta (EO)',
        'Gamma (EO)',
        'Theta (EC)',
        'Alpha (EC)',
        'Beta (EC)',
        'Gamma (EC)',
    ],
)
df_X.index = df_y.index
df_all = pd.concat([df_X, df_y], axis=1)
df_all

# %%
plt.figure()
sns.heatmap(df_all.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix Heatmap')
plt.show()

# %%
