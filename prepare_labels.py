# %%
import os
import sys

import pandas as pd

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

# Dump the labels to pickle files
data_loader = DataLoader(os.path.join(os.getcwd(), 'data'))
data_loader.save_pkl(df_psq['PSQ_OverallScore'], os.path.join('labels', 'y_psq'))
data_loader.save_pkl(df_tas['TAS_OverallScore'], os.path.join('labels', 'y_tas'))
data_loader.save_pkl(df_tic['TICS_ScreeningScale'], os.path.join('labels', 'y_tic'))

# %%
