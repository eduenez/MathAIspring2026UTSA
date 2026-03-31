import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from ipywidgets import interact, IntSlider, Button, VBox, Output

# --- Configuration ---
N_SAMPLES = 20
NOISE_LEVEL = 0.4
MAX_DEGREE = 25

# --- Generate Data ---
def generate_data(n_samples, noise):
    # The true function is a sine wave
    X = np.linspace(0, 2 * np.pi, n_samples)
    y_true = np.sin(X)
    # Add Gaussian noise
    y_noisy = y_true + np.random.normal(0, noise, n_samples)
    return X, y_true, y_noisy

# Initialize global state (to keep true function consistent across updates)
X, y_true, y_noisy = generate_data(N_SAMPLES, NOISE_LEVEL)
x_plot = np.linspace(0, 2 * np.pi, 200) # Smooth x for plotting

out = Output()

def plot_interactive(degree):
    global X, y_true, y_noisy

    with out:
        out.clear_output(wait=True)
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot true function and current training data
        ax.plot(x_plot, np.sin(x_plot), label='True Function ($f(x) = sin(x)$)', color='gray', linestyle='--')
        ax.scatter(X, y_noisy, label='Training Data (with noise)', edgecolors='k')

        # Fit polynomial model
        model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
        model.fit(X.reshape(-1, 1), y_noisy)
        y_pred = model.predict(x_plot.reshape(-1, 1))

        # Calculate fit metrics (approximate indicators)
        train_mse = np.mean((model.predict(X.reshape(-1, 1)) - y_noisy)**2)
        # In a real scenario, bias and variance require multiple datasets.
        # Here we use heuristic proxies for illustration:
        # High Bias proxy: Large gap between true function and model even with low noise.
        # High Variance proxy: Large wild oscillations in predictions.

        # Plot the fitted model
        ax.plot(x_plot, y_pred, color='red', label=f'Model (Degree {degree})')

        ax.set_ylim(-2.5, 2.5)
        ax.set_title(f'Polynomial Fit (Degree {degree})')
        ax.legend()
        plt.show()

def on_generate_clicked(b):
    global X, y_true, y_noisy
    # Regenerate training points but keep the original true function
    X, _, y_noisy = generate_data(N_SAMPLES, NOISE_LEVEL)
    plot_interactive(degree_slider.value)

# Define Slider and Button
degree_slider = IntSlider(value=1, min=1, max=MAX_DEGREE, step=1, description='Polynomial Degree')
generate_btn = Button(description='Generate New Data')
generate_btn.on_click(on_generate_clicked)

# Wire up interactivity
interact(plot_interactive, degree=degree_slider)
display(VBox([generate_btn, out]))