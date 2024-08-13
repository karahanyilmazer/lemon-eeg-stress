# %%
#!%matplotlib qt
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots

plt.style.use(['science', 'no-latex'])
plt.matplotlib.rcParams['figure.dpi'] = 260
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
    df = df[df['Label'].str.contains('PSQ')]
    df = df.sort_values('R^2 (Test)')
    model_name.append(file.split('.')[0].split('_'))
    psq_test.append(df[df['Label'] == label]['R^2 (Test)'].values[0])
    psq_train.append(df[df['Label'] == label]['R^2 (Train)'].values[0])
    psq_train_mean.append(df[df['Label'] == label]['R^2 Mean (Train)'].values[0])
    psq_cv_mean.append(df[df['Label'] == label]['R^2 Mean (CV)'].values[0])
    psq_train_std.append(df[df['Label'] == label]['R^2 STD (Train)'].values[0])
    psq_cv_std.append(df[df['Label'] == label]['R^2 STD (CV)'].values[0])

df = pd.read_csv('eegnet_result.csv')
model_name.append('Modified EEGNet')
psq_cv_mean.append(df['r2'].iloc[5:].mean())
psq_cv_std.append(df['r2'].iloc[5:].std())
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

res_df = pd.DataFrame([psq_cv_mean, psq_cv_std, psq_test]).T
res_df.index = transformed_model_names
res_df.columns = ['R2 Mean (CV)', 'R2 STD (CV)', 'R2 (Test)']
res_df = res_df.sort_values('R2 (Test)', ascending=True)
res_df
# %%
bar_width = 0.3
r1 = np.arange(len(transformed_model_names))
r2 = [x + bar_width for x in r1]

plt.figure(figsize=(10, 5))
plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
plt.bar(
    r1,
    res_df['R2 Mean (CV)'],
    width=bar_width,
    label='Validation',
    yerr=res_df['R2 STD (CV)'],
    capsize=2,
)
plt.bar(r2, res_df['R2 (Test)'], width=bar_width, label='Test')

plt.title('$R^2$ Score of All Feature and Model Combinations')
plt.xlabel('Model')
plt.ylabel('$R^2$ Score')
plt.xticks([r + bar_width / 2 for r in r1], transformed_model_names, rotation=60)
plt.ylim(-0.37, 1)

plt.legend()
plt.savefig('img/r2_of_all_models.png', dpi=300, transparent=False)
plt.show()


# %%
bar_width = 0.3
tmp_names = transformed_model_names.copy()[:-1] + ['']
tmp_mean_cv = res_df['R2 Mean (CV)']
tmp_mean_cv['Modified EEGNet'] = 0
tmp_std_cv = res_df['R2 STD (CV)']
tmp_std_cv['Modified EEGNet'] = 0
tmp_test = res_df['R2 (Test)']
tmp_test['Modified EEGNet'] = 0
r1 = np.arange(len(transformed_model_names))
r2 = [x + bar_width for x in r1]

plt.figure(figsize=(10, 5))
plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
plt.bar(
    r1,
    tmp_mean_cv,
    width=bar_width,
    label='Validation',
    yerr=tmp_std_cv,
    capsize=2,
)
plt.bar(r2, tmp_test, width=bar_width, label='Test')

plt.title('$R^2$ Score of All Feature and Model Combinations')
plt.xlabel('Model')
plt.ylabel('$R^2$ Score')
plt.xticks([r + bar_width / 2 for r in r1], tmp_names, rotation=60)
plt.ylim(-0.37, 1)

plt.legend()
plt.savefig('img/r2_of_all_models_without_eegnet.png', dpi=300, transparent=False)
plt.show()

# %%
# Plot the distributon of the TAS_OverallScore
# Load the data
test = 'PSQ'
y = pd.read_csv(
    f'data/behavioral/Emotion_and_Personality_Test_Battery_LEMON/{test}.csv'
)
# Convert the values to numeric
plt.figure()
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
df_r2 = pd.read_csv('eegnet_result.csv')

plt.figure()
plt.plot(df_r2['loss'], label='Training Loss')
plt.plot(df_r2['val_loss'], label='Validation Loss')
plt.plot(df_r2['r2'], label='Validation R^2')
plt.title('Loss of Modified EEGNet')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.savefig('img/eegnet_loss.png', dpi=300, transparent=False)
plt.show()
# %%

# Load your data
df_r2 = pd.read_csv('eegnet_result.csv')

# Create a figure and axis
fig, ax1 = plt.subplots()

# Plot training and validation loss on the first y-axis
ax1.plot(df_r2['loss'], label='Training Loss', color='blue')
ax1.plot(df_r2['val_loss'], label='Validation Loss', color='green')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('MSE Loss', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

# Create a second y-axis for the validation R^2
ax2 = ax1.twinx()
ax2.plot(df_r2['r2'], label='Validation $R^2$', color='red')
ax2.set_ylabel('$R^2$ Score', color='red')
ax2.tick_params(axis='y', labelcolor='red')

# Title and legend
plt.title('EEGNet Training Over Epochs')

# To handle legend for both axes
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc='center right')

# Save the plot
plt.savefig('img/eegnet_training.png', dpi=300, transparent=False)

# Show the plot
plt.show()

# %%
