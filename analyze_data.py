# %%
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from tqdm import tqdm

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

# Get a list of all subjects with EEG data
subj_list = os.listdir(os.path.join(os.getcwd(), 'data', 'EEG_preprocessed'))

# Read all possible labels
df_y = pd.DataFrame()
for file in os.listdir(data_dir):
    if file.endswith('.csv'):
        # Skip the YFAS file with missing data
        if 'YFAS' in file:
            continue
        df = pd.read_csv(os.path.join(data_dir, file), index_col=0).sort_index()
        df_y = pd.concat([df_y, df], axis=1)

# Filter columns that contain 'TICS' in their name
tics_columns = [col for col in df_y.columns if 'TICS' in col]

# Sum the selected columns row-wise and assign to a new column
df_y['TICS_OverallScore'] = df_y[tics_columns].sum(axis=1)

bad_subjs = ['sub-032345', 'sub-032357', 'sub-032450', 'sub-032493', 'sub-032513']

# Clean the NaN values
y = df_y[df_y.index.isin(subj_list)]
y = y.dropna(axis=1, thresh=y.shape[0] - 9)
y = y.dropna(axis=0, thresh=y.shape[1] - 8)
y = y.ffill()
y = y.drop(bad_subjs)

# Calculate the dropped indices
dropped_subjects = df_y.index.difference(y.index)
dropped_subjects = dropped_subjects.union(bad_subjs)

# Load the data
data_loader = DataLoader(os.path.join(os.getcwd(), 'data'))
X_bp_abs = data_loader.load_pkl(os.path.join('feat_mats', 'X_bp_abs_interp'))
X_bp_rel = data_loader.load_pkl(os.path.join('feat_mats', 'X_bp_rel_interp'))
X_welch = data_loader.load_pkl(os.path.join('feat_mats', 'X_welch_interp'))
X_var = data_loader.load_pkl(os.path.join('feat_mats', 'X_var_interp'))
X_logvar = data_loader.load_pkl(os.path.join('feat_mats', 'X_logvar_interp'))

# %%
# Define the feature and label dictionaries
feat_dict = {
    'X_bp_rel': X_bp_rel,
    # 'X_bp_abs': X_bp_abs,
    'X_welch': X_welch,
    'X_var': X_var,
    'X_logvar': X_logvar,
}

for selected_feat in feat_dict.keys():
    # Pick the features to use
    print('selected_feat: ', selected_feat)
    X = pd.DataFrame(
        feat_dict[selected_feat],
        index=os.listdir(os.path.join(os.getcwd(), 'data', 'EEG_preprocessed')),
    )
    X = X.drop(dropped_subjects, errors='ignore')

    # Initialize KFold with 3 splits
    kf = KFold(n_splits=3, shuffle=True)

    svr_params = {
        'kernel': 'rbf',
    }

    rf_params = {
        'n_estimators': 200,
        'max_depth': 4,
        'criterion': 'absolute_error',
    }

    xg_boost_params = {
        'n_estimators': 200,
        'max_depth': 4,
        'learning_rate': 0.01,
        'loss': 'absolute_error',
    }

    # Create a pipeline
    pipe1 = Pipeline([('scaler', StandardScaler()), ('svr', SVR(**svr_params))])
    pipe2 = Pipeline(
        [('scaler', StandardScaler()), ('rf', RandomForestRegressor(**rf_params))]
    )
    pipe3 = Pipeline(
        [
            ('scaler', StandardScaler()),
            ('xg', GradientBoostingRegressor(**xg_boost_params)),
        ]
    )

    pipes = [pipe1, pipe2, pipe3]

    for pipe in pipes:
        print('pipe:', pipe[-1])

        r2_train_mean = []
        r2_test_mean = []
        mae_train_mean = []
        mae_test_mean = []
        r2_train_std = []
        r2_test_std = []
        mae_train_std = []
        mae_test_std = []
        r2_train = []
        r2_test = []
        mae_train = []
        mae_test = []
        labels = []

        # Iterate over the columns of the DataFrame
        for selected_label in tqdm(y.columns):
            r2_train_tmp = []
            r2_test_tmp = []
            mae_train_tmp = []
            mae_test_tmp = []

            y_tmp = y[selected_label]

            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_tmp, train_size=0.8
            )

            for train_index, test_index in kf.split(X_train):
                X_cv_train, X_cv_test = (
                    X_train.iloc[train_index],
                    X_train.iloc[test_index],
                )
                y_cv_train, y_cv_test = (
                    y_train.iloc[train_index],
                    y_train.iloc[test_index],
                )

                # Fit the whole training set
                pipe.fit(X_cv_train, y_cv_train)
                y_pred_train = pipe.predict(X_cv_train)
                y_pred_test = pipe.predict(X_cv_test)

                # Store the values for later
                r2_train_tmp.append(r2_score(y_cv_train, y_pred_train))
                r2_test_tmp.append(r2_score(y_cv_test, y_pred_test))
                mae_train_tmp.append(mean_absolute_error(y_cv_train, y_pred_train))
                mae_test_tmp.append(mean_absolute_error(y_cv_test, y_pred_test))

            labels.append(selected_label)
            r2_train_mean.append(np.mean(r2_train_tmp))
            r2_test_mean.append(np.mean(r2_test_tmp))
            mae_train_mean.append(np.mean(mae_train_tmp))
            mae_test_mean.append(np.mean(mae_test_tmp))
            r2_train_std.append(np.std(r2_train_tmp))
            r2_test_std.append(np.std(r2_test_tmp))
            mae_train_std.append(np.std(mae_train_tmp))
            mae_test_std.append(np.std(mae_test_tmp))

            pipe.fit(X_train, y_train)
            y_pred_train = pipe.predict(X_train)
            y_pred_test = pipe.predict(X_test)

            r2_train.append(r2_score(y_train, y_pred_train))
            r2_test.append(r2_score(y_test, y_pred_test))

        # Print the results
        result_df = pd.DataFrame(
            {
                'Label': labels,
                'R^2 Mean (Train)': r2_train_mean,
                'R^2 Mean (CV)': r2_test_mean,
                'MAE Mean (Train)': mae_train_mean,
                'MAE Mean (CV)': mae_test_mean,
                'R^2 STD (Train)': r2_train_std,
                'R^2 STD (CV)': r2_test_std,
                'MAE STD (Train)': mae_train_std,
                'MAE STD (CV)': mae_test_std,
                'R^2 (Train)': r2_train,
                'R^2 (Test)': r2_test,
            }
        )
        # result_df = result_df.sort_values('R^2 (CV)', ascending=False)
        result_df = result_df.sort_values('MAE (CV)', ascending=True)
        result_df.to_csv(
            os.path.join('results', f'{selected_feat}_{pipe.steps[-1][0]}.csv')
        )

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
