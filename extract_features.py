# %%
import os
import sys

import matplotlib.pyplot as plt
import mne
import numpy as np

sys.path.append(os.path.join(os.getcwd(), '..', 'eeg-classes'))
from src.preprocessing.FeatureExtractor import FeatureExtractor
from src.utils.DataLoader import DataLoader

# %%
# Get the data directory
base_dir = os.path.join(os.getcwd(), 'data', 'EEG')
# Get a list of all the subjects
subjects = os.listdir(base_dir)

# Initialize the data loader
loader = DataLoader()

# Initialize the feature matrices
n_subs_per_mat = len(subjects)
n_feats_bp = 4
n_feats_welch = 60
n_feats_var = 60
X_bp_abs = np.zeros((n_subs_per_mat, 2 * n_feats_bp))
X_bp_rel = np.zeros((n_subs_per_mat, 2 * n_feats_bp))
X_welch = np.zeros((n_subs_per_mat, 2 * n_feats_welch))
X_var = np.zeros((n_subs_per_mat, 2 * n_feats_var))

# Define the frequency bands of interest
freq_bands = ['theta', 'alpha', 'beta', 'low_gamma']


for i, subj in enumerate(subjects):
    # EYES OPEN
    # ==================================================================================
    # Read in the data
    eo_path = os.path.join(base_dir, subj, f'{subj}_EO.set')
    raw_eo = mne.io.read_raw_eeglab(eo_path)

    # Set necessary variables for feature extraction
    info = raw_eo.info
    ch_names = raw_eo.ch_names
    data = raw_eo.get_data()

    # Extract features
    extractor = FeatureExtractor(info)
    X_bp_abs[i, n_feats_bp:] = extractor.get_bp_feat(
        data,
        ch_names,
        freq_bands=freq_bands,
        relative=False,
    ).ravel()
    X_bp_rel[i, n_feats_bp:] = extractor.get_bp_feat(
        data,
        ch_names,
        freq_bands=freq_bands,
        relative=True,
    ).ravel()
    # X_welch[i, n_feats_welch:] = extractor.get_welch_feat(data, ch_names).ravel()
    # X_var[i, n_feats_var:] = extractor.get_var_feat(data, ch_names).ravel()

    # EYES CLOSED
    # ==================================================================================
    # Read in the data
    ec_path = os.path.join(base_dir, subj, f'{subj}_EC.set')
    raw_ec = mne.io.read_raw_eeglab(ec_path)

    # Set necessary variables for feature extraction
    info = raw_ec.info
    ch_names = raw_ec.ch_names
    data = raw_ec.get_data()

    # Extract features
    extractor = FeatureExtractor(info)
    X_bp_abs[i, :n_feats_bp] = extractor.get_bp_feat(
        data,
        ch_names,
        freq_bands=freq_bands,
        relative=False,
    ).ravel()
    X_bp_rel[i, :n_feats_bp] = extractor.get_bp_feat(
        data,
        ch_names,
        freq_bands=freq_bands,
        relative=True,
    ).ravel()
    # X_welch[i, :n_feats_welch] = extractor.get_welch_feat(data, ch_names).ravel()
    # X_var[i, :n_feats_var] = extractor.get_var_feat(data, ch_names).ravel()

# %%
# Save the feature matrices
loader.save_pkl(
    X_bp_abs,
    os.path.join(
        'feat_mats',
        f'X_test_bp_abs',
    ),
)
loader.save_pkl(
    X_bp_rel,
    os.path.join(
        'feat_mats',
        f'X_test_bp_rel',
    ),
)

# %%
