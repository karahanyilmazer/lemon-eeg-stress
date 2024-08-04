# %%
import os
import sys

import mne
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'eeg-classes'))
from src.preprocessing.FeatureExtractor import FeatureExtractor  # type: ignore
from src.utils.DataLoader import DataLoader  # type: ignore

# %%
mne.set_log_level('ERROR')

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

# Get the data directory
base_dir = os.path.join(os.getcwd(), 'data', 'EEG_preprocessed')
# Get a list of all the subjects
subjects = os.listdir(base_dir)

# Initialize the data loader
loader = DataLoader(os.path.join(os.getcwd(), 'data'))

# Initialize the feature matrices
n_subs_per_mat = len(subjects)
n_feats_bp = 3
n_feats_welch = len(all_chans)
n_feats_var = len(all_chans)
n_feats_logvar = len(all_chans)
X_bp_abs = np.zeros((n_subs_per_mat, 2 * n_feats_bp))
X_bp_rel = np.zeros((n_subs_per_mat, 2 * n_feats_bp))
X_welch = np.zeros((n_subs_per_mat, 2 * n_feats_welch))
X_var = np.zeros((n_subs_per_mat, 2 * n_feats_var))
X_logvar = np.zeros((n_subs_per_mat, 2 * n_feats_logvar))

# Define the frequency bands of interest
freq_bands = ['theta', 'alpha', 'beta']
subj_list = []


for i, subj in tqdm(enumerate(subjects), total=len(subjects)):
    # EYES OPEN
    # ==================================================================================
    # Read in the data
    eo_path = os.path.join(base_dir, subj, f'{subj}_EO.set')
    raw_eo = mne.io.read_raw_eeglab(eo_path, preload=True)

    # Add and interpolate missing channels
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
    X_welch[i, n_feats_welch:] = extractor.get_welch_feat(data, ch_names).ravel()
    X_var[i, n_feats_var:] = extractor.get_var_feat(data, ch_names).ravel()
    X_logvar[i, n_feats_var:] = extractor.get_logvar_feat(data, ch_names).ravel()

    # EYES CLOSED
    # ==================================================================================
    # Read in the data
    ec_path = os.path.join(base_dir, subj, f'{subj}_EC.set')
    raw_ec = mne.io.read_raw_eeglab(ec_path, preload=True)

    # Add and interpolate missing channels
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
    raw_ec = raw_ec.interpolate_bads(reset_bads=True)

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
    X_welch[i, :n_feats_welch] = extractor.get_welch_feat(data, ch_names).ravel()
    X_var[i, :n_feats_var] = extractor.get_var_feat(data, ch_names).ravel()
    X_logvar[i, :n_feats_var] = extractor.get_logvar_feat(data, ch_names).ravel()

    subj_list.append(subj)

# %%
# Save the feature matrices
loader.save_pkl(
    X_bp_abs,
    os.path.join(
        'feat_mats',
        'X_bp_abs_interp',
    ),
)
loader.save_pkl(
    X_bp_rel,
    os.path.join(
        'feat_mats',
        'X_bp_rel_interp',
    ),
)
loader.save_pkl(
    X_welch,
    os.path.join(
        'feat_mats',
        'X_welch_interp',
    ),
)
loader.save_pkl(
    X_var,
    os.path.join(
        'feat_mats',
        'X_var_interp',
    ),
)
loader.save_pkl(
    X_logvar,
    os.path.join(
        'feat_mats',
        'X_logvar_interp',
    ),
)

# %%
