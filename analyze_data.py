# %%
#!%matplotlib qt
import os
import sys

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

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
X_bp_abs = data_loader.load_pkl(os.path.join('feat_mats', 'X_bp_abs_new'))
X_bp_rel = data_loader.load_pkl(os.path.join('feat_mats', 'X_bp_rel_new'))

# %%
# Define the feature and label dictionaries
feat_dict = {'X_bp_rel': X_bp_rel, 'X_bp_abs': X_bp_abs}

# Pick the features to use
selected_feat = 'X_bp_rel'
X = feat_dict[selected_feat]

r2_train = []
r2_test = []
r2_train_tmp = []
r2_test_tmp = []
mae_train = []
mae_test = []
mae_train_tmp = []
mae_test_tmp = []
labels = []

# Initialize KFold with 5 splits
kf = KFold(n_splits=5, shuffle=True)

kernel = 'linear'
degree = 3

# Create a pipeline
pipe = Pipeline([('scaler', StandardScaler()), ('svr', SVR(kernel=kernel))])

# Iterate over the DataFrames
for df in [df_psq, df_tas, df_tic]:
    # Iterate over the columns of the DataFrame
    for selected_label in df.columns:
        y = df[selected_label]

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

        for train_index, test_index in kf.split(X_train):
            X_cv_train, X_cv_test = X_train[train_index], X_train[test_index]
            y_cv_train, y_cv_test = y_train.iloc[train_index], y_train.iloc[test_index]

            # Fit the whole training set
            pipe.fit(X_cv_train, y_cv_train)
            y_pred_train = pipe.predict(X_cv_train)
            y_pred_test = pipe.predict(X_cv_test)

            # Store the values for later
            r2_train_tmp.append(r2_score(y_cv_train, y_pred_train))
            r2_test_tmp.append(r2_score(y_cv_test, y_pred_test))
            mae_train_tmp.append(mean_absolute_error(y_cv_train, y_pred_train))
            mae_test_tmp.append(mean_absolute_error(y_cv_test, y_pred_test))

        r2_train.append(np.mean(r2_train_tmp))
        r2_test.append(np.mean(r2_test_tmp))
        mae_train.append(np.mean(mae_train_tmp))
        mae_test.append(np.mean(mae_test_tmp))
        labels.append(selected_label)

# Print the results
result_df = pd.DataFrame(
    {
        'Label': labels,
        'R^2 (Train)': r2_train,
        'R^2 (CV)': r2_test,
        'MAE (Train)': mae_train,
        'MAE (CV)': mae_test,
    }
)
# result_df = result_df.sort_values('R^2 (CV)', ascending=False)
result_df = result_df.sort_values('MAE (CV)', ascending=True)
result_df

# %%
