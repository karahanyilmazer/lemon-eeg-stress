# %%
import os
import sys

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'eeg-classes'))
from src.utils.DataLoader import DataLoader  # type: ignore

# %%
# Load the data
data_loader = DataLoader(os.path.join(os.getcwd(), 'data'))
X_bp_abs = data_loader.load_pkl(os.path.join('feat_mats', 'X_bp_abs'))
X_bp_rel = data_loader.load_pkl(os.path.join('feat_mats', 'X_bp_rel'))

# Load the labels
y_psq = data_loader.load_pkl(os.path.join('labels', 'y_psq'))
y_tas = data_loader.load_pkl(os.path.join('labels', 'y_tas'))
y_tic = data_loader.load_pkl(os.path.join('labels', 'y_tic'))

# %%
# Define the feature and label dictionaries
feat_dict = {'X_bp_rel': X_bp_rel, 'X_bp_abs': X_bp_abs}
label_dict = {'y_psq': y_psq, 'y_tas': y_tas, 'y_tic': y_tic}

# Pick the features to use
selected_feat = 'X_bp_rel'
selected_label = 'y_tic'

X = feat_dict[selected_feat]
y = label_dict[selected_label]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

# Create a pipeline
pipe = Pipeline([('scaler', StandardScaler()), ('pca', PCA()), ('svr', SVR())])
pipe.fit(X_train, y_train)

acc_score = pipe.score(X_train, y_train)

# Print the results
print(f'Feature matrix:\t{selected_feat}')
print(f'Labels:\t\t{selected_label}\n')
print('R^2 on training data:', acc_score)

# %%
