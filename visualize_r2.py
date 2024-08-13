# %%
#!%matplotlib qt
import matplotlib.pyplot as plt
import numpy as np
import scienceplots
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures

plt.style.use(['science', 'no-latex'])
plt.matplotlib.rcParams['figure.dpi'] = 300
plt.matplotlib.rcParams['xtick.minor.visible'] = False

# Set seed for reproducibility
np.random.seed(42)

# %%
# Create a dataset with R2 = 0 (random scatter)
# Generate data from a sine wave with noise
x = np.linspace(0, 10, 100).reshape(-1, 1)
y = np.sin(x) + np.random.normal(0, 0.1, x.shape)

# Use a polynomial model to fit the data (higher degree than necessary)
degree = 10  # Deliberately using a high degree polynomial
poly_features = PolynomialFeatures(degree=degree)
x_poly = poly_features.fit_transform(x)

# Fit the polynomial regression model
model = LinearRegression().fit(x_poly, y)
y_pred_bad = np.zeros_like(y)  # Predictions are zero
y_pred_good = model.predict(x_poly)

# Calculate R^2
r2_bad = r2_score(y, y_pred_bad)
r2_good = r2_score(y, y_pred_good)

# Plotting the figures
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot for bad fit
axes[0].scatter(x, y, label='Data')
axes[0].plot(x, y_pred_bad, color='red', label='Model', lw=2)
axes[0].set_title(f'Bad Fit ($R^2 = {r2_bad:.2f}$)', fontsize=17)
axes[0].set_xlabel('x', fontsize=15)
axes[0].set_ylabel('y', fontsize=15)
axes[0].set_xticklabels([])
axes[0].set_yticklabels([])
axes[0].legend(loc='lower left')

# Plot for good fit
axes[1].scatter(x, y, label='Data')
axes[1].plot(x, y_pred_good, color='red', label='Model', lw=2)
axes[1].set_title(f'Good Fit ($R^2 = {r2_good:.2f}$)', fontsize=17)
axes[1].set_xlabel('x', fontsize=15)
# axes[1].set_ylabel('y')
axes[1].set_xticklabels([])
axes[1].set_yticklabels([])
axes[1].legend(loc='lower left')

plt.tight_layout()
plt.savefig('r2_visualization.png')
plt.show()

# %%
