import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

st.set_page_config(page_title="Linear Regression Visualizer", layout="wide")

st.title("📈 Linear Regression Interactive Visualizer")

# ---------------------------
# Sidebar Controls
# ---------------------------
st.sidebar.header("⚙️ Controls")

noise = st.sidebar.slider("Noise Level", 0.0, 5.0, 1.0)
num_points = st.sidebar.slider("Data Points", 10, 300, 50, 10)
m = st.sidebar.slider("Slope (m)", -5.0, 5.0, 0.0)
b = st.sidebar.slider("Intercept (b)", -10.0, 10.0, 0.0)

learning_rate = st.sidebar.slider("Learning Rate", 0.001, 0.1, 0.01)
iterations = st.sidebar.slider("Iterations", 10, 200, 50)

add_outliers = st.sidebar.checkbox("Add Outliers")
run_gd = st.sidebar.checkbox("Run Gradient Descent")

# ---------------------------
# Data Generation
# ---------------------------
np.random.seed(42)
X = np.linspace(0, 10, num_points)
y = 2 * X + 3 + np.random.randn(num_points) * noise

if add_outliers:
    X = np.append(X, [2, 8])
    y = np.append(y, [30, -10])

# Predictions
y_pred = m * X + b

# ---------------------------
# Tabs Layout
# ---------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Data & Fit",
    "📉 Error",
    "🗻 Loss Surface",
    "🔄 Gradient Descent",
    "⚡ Learning Rate"
])

# ---------------------------
# TAB 1: DATA & FIT
# ---------------------------
with tab1:
    st.subheader("📊 Data & Line Fit")
    st.info("👉 Adjust slope and intercept to see how the line fits the data.")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(X, y, label="Data")
    ax.plot(X, y_pred, color="red", label="Model")
    ax.legend()
    st.pyplot(fig)

# ---------------------------
# TAB 2: ERROR VISUALIZATION
# ---------------------------
with tab2:
    st.subheader("📉 Error / Residuals")

    mse = np.mean((y - y_pred) ** 2)
    st.write(f"### MSE: {mse:.4f}")
    st.info("👉 Vertical lines show errors (residuals). Larger lines = larger error.")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(X, y)
    ax.plot(X, y_pred, color="red")

    for i in range(len(X)):
        ax.plot([X[i], X[i]], [y[i], y_pred[i]], color='gray')

    st.pyplot(fig)

# ---------------------------
# TAB 3: LOSS SURFACE
# ---------------------------
with tab3:
    st.subheader("🗻 Loss Landscape")
    st.info("👉 Lowest point represents best slope and intercept.")

    m_vals = np.linspace(-3, 5, 50)
    b_vals = np.linspace(-5, 10, 50)

    M, B = np.meshgrid(m_vals, b_vals)
    Z = np.zeros_like(M)

    for i in range(len(m_vals)):
        for j in range(len(b_vals)):
            y_temp = M[i, j] * X + B[i, j]
            Z[i, j] = np.mean((y - y_temp) ** 2)

    fig = go.Figure(data=go.Contour(
        z=Z,
        x=m_vals,
        y=b_vals,
        colorscale='Viridis'
    ))

    fig.update_layout(
        title="Loss Contour",
        xaxis_title="Slope (m)",
        yaxis_title="Intercept (b)",
        height=380,
    )
    st.plotly_chart(fig)

# ---------------------------
# TAB 4: GRADIENT DESCENT
# ---------------------------
with tab4:
    st.subheader("🔄 Gradient Descent")
    st.info("👉 Watch how parameters move toward minimum error.")

    if run_gd:
        m_gd, b_gd = 0, 0
        history = []

        for _ in range(iterations):
            y_gd = m_gd * X + b_gd

            dm = -2 * np.mean(X * (y - y_gd))
            db = -2 * np.mean(y - y_gd)

            m_gd -= learning_rate * dm
            b_gd -= learning_rate * db

            history.append((m_gd, b_gd))

        st.write(f"Final m: {m_gd:.2f}, b: {b_gd:.2f}")

        # Plot path on contour
        path_m = [p[0] for p in history]
        path_b = [p[1] for p in history]

        fig = go.Figure(data=go.Contour(
            z=Z,
            x=m_vals,
            y=b_vals,
            colorscale='Viridis'
        ))

        fig.add_trace(go.Scatter(
            x=path_m,
            y=path_b,
            mode='lines+markers',
            name='Gradient Path',
            line=dict(color='red')
        ))

        fig.update_layout(title="Gradient Descent Path", height=380)
        st.plotly_chart(fig)
    else:
        st.warning("Enable 'Run Gradient Descent' from sidebar")

# ---------------------------
# TAB 5: LEARNING RATE
# ---------------------------
with tab5:
    st.subheader("⚡ Learning Rate Comparison")
    st.info("👉 Compare how different learning rates affect convergence.")

    def run_lr(lr):
        m_lr, b_lr = 0, 0
        losses = []

        for _ in range(50):
            y_lr = m_lr * X + b_lr
            loss = np.mean((y - y_lr) ** 2)
            losses.append(loss)

            dm = -2 * np.mean(X * (y - y_lr))
            db = -2 * np.mean(y - y_lr)

            m_lr -= lr * dm
            b_lr -= lr * db

        return losses

    lr_small = run_lr(0.001)
    lr_good = run_lr(0.01)
    lr_large = run_lr(0.1)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(lr_small, label="Small LR (Slow)")
    ax.plot(lr_good, label="Optimal LR")
    ax.plot(lr_large, label="Large LR (Unstable)")
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Loss")
    ax.legend()

    st.pyplot(fig)