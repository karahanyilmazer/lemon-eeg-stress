# %%
#!%matplotlib qt
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots

plt.style.use(['science', 'no-latex'])
# plt.matplotlib.rcParams['figure.dpi'] = 260
plt.matplotlib.rcParams['xtick.minor.visible'] = False

# %%
label = ['PSQ_OverallScore', 'TAS_OverallScore'][0]

model_name = []
psq_train_mean = []
psq_train_std = []
psq_cv_mean = []
psq_cv_std = []
psq_train = []
psq_test = []

for file in os.listdir('results'):
    df = pd.read_csv(f'results/{file}')
    # df = df[df['Label'].str.contains('PSQ|TICS')]
    model_name.append(file.split('.')[0].split('_'))
    psq_test.append(df[df['Label'] == label]['R^2 (Test)'].values[0])
    psq_train.append(df[df['Label'] == label]['R^2 (Train)'].values[0])
    psq_train_mean.append(df[df['Label'] == label]['R^2 Mean (Train)'].values[0])
    psq_cv_mean.append(df[df['Label'] == label]['R^2 Mean (CV)'].values[0])
    psq_train_std.append(df[df['Label'] == label]['R^2 STD (Train)'].values[0])
    psq_cv_std.append(df[df['Label'] == label]['R^2 STD (CV)'].values[0])

df = pd.read_csv('eegnet_result.csv')
model_name.append('Modified EEGNet')
psq_cv_mean.append(df['r2'].mean())
psq_cv_std.append(df['r2'].std())
psq_test.append(0.5943)

# Mapping of abbreviations to full names
name_mapping = {
    'logvar': 'Log-Var',
    'bp': 'BP',
    'rel': 'Rel',
    'xg': 'XGBoost',
    'svr': 'SVR',
    'var': 'Var',
    'rf': 'RF',
    'welch': 'Welch',
    'Modified EEGNet': 'Modified EEGNet',
}

# Transform model names
transformed_model_names = []
for name in model_name:
    if isinstance(name, list):
        transformed_name = ' + '.join(
            [name_mapping[part] for part in name if part in name_mapping]
        )
        # Special case for 'bp rel' to 'Rel. BP'
        if 'BP' in transformed_name and 'Rel' in transformed_name:
            transformed_name = transformed_name.replace('BP + Rel', 'Rel. BP')
    else:
        transformed_name = name_mapping.get(name, name)
    transformed_model_names.append(transformed_name)

# %%
bar_width = 0.3
r1 = np.arange(len(transformed_model_names))
r2 = [x + bar_width for x in r1]

plt.bar(
    r1,
    psq_cv_mean,
    width=bar_width,
    label='Validation',
    yerr=psq_cv_std,
    capsize=2,
)
plt.bar(r2, psq_test, width=bar_width, label='Test')

plt.xlabel('Model')
plt.ylabel('$R^2 Score$')
plt.xticks([r + bar_width / 2 for r in r1], transformed_model_names, rotation=45)
plt.ylim(plt.ylim()[0], 1)

plt.legend()
plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
plt.show()

# %%
# Plot the distributon of the TAS_OverallScore
# Load the data
test = 'PSQ'
y = pd.read_csv(
    f'data/behavioral/Emotion_and_Personality_Test_Battery_LEMON/{test}.csv'
)
# Convert the values to numeric
y[f'{test}_OverallScore'] = pd.to_numeric(y[f'{test}_OverallScore'], errors='coerce')
y[f'{test}_OverallScore'].plot.hist(bins=20)
plt.ylabel('Frequency')
plt.xlabel(f'{test}_OverallScore')
plt.show()

print(
    y[f'{test}_OverallScore'].min(),
    y[f'{test}_OverallScore'].max(),
    y[f'{test}_OverallScore'].mean(),
    y[f'{test}_OverallScore'].std(),
)
# %%
first_color = '#0C5DA5'
second_color = '#00B945'
# %%
