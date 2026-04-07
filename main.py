import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("📈 Linear Regression Visualizer")

# Generate synthetic data
np.random.seed(42)
X = np.linspace(0, 10, 50)
noise = st.sidebar.slider("Noise Level", 0.0, 5.0, 1.0)
y = 2 * X + 3 + np.random.randn(50) * noise

# Sliders for line
m = st.sidebar.slider("Slope (m)", -5.0, 5.0, 0.0)
b = st.sidebar.slider("Intercept (b)", -10.0, 10.0, 0.0)

# Prediction
y_pred = m * X + b

# Plot
fig, ax = plt.subplots()
ax.scatter(X, y, label="Data")
ax.plot(X, y_pred, color="red", label="Model")
ax.legend()
# st.pyplot(fig)
mse = np.mean((y - y_pred) ** 2)
st.write(f"📉 MSE: {mse:.4f}")

# Residuals
fig, ax = plt.subplots()
ax.scatter(X, y)
ax.plot(X, y_pred, color="red")

for i in range(len(X)):
    ax.plot([X[i], X[i]], [y[i], y_pred[i]], color='gray')

st.pyplot(fig)