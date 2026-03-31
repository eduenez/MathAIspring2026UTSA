import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import FloatSlider, VBox, HTML, interactive_output, Layout

# 1. Define the x-axis (Model Complexity from 0 to 100)
x = np.linspace(0, 100, 500)

# 2. Define the mathematical functions for the curves
# Bias^2 decreases as complexity increases
bias_squared = 8000 / (x + 20)**1.5

# Variance increases as complexity increases
variance = 0.05 * x**1.8

# Irreducible error is a constant baseline noise
irreducible_error = np.full_like(x, 15)

# Total Test Error is the sum of the components (The U-Shape)
test_error = bias_squared + variance + irreducible_error

# Training error continuously decreases
train_error = 7000 / (x + 20)**1.6

# Highlight the optimal complexity point (minimum test error)
optimal_idx = np.argmin(test_error)
optimal_x = x[optimal_idx]

def plot_interactive(complexity):
    # 3. Setup the figure and main axis
    fig, ax = plt.subplots(figsize=(10, 6))
    # Tight margins so plot area spans nearly the full figure width,
    # matching the slider track for vertical alignment of knob and line.
    fig.subplots_adjust(left=0.065, right=0.985, top=0.95, bottom=0.07)

    # Plot the static curves
    ax.plot(x, bias_squared, label='Bias²', color='#1f77b4', linestyle='--')
    ax.plot(x, variance, label='Variance', color='#2ca02c', linestyle='--')
    ax.plot(x, train_error, label='Training Error', color='#ff7f0e')
    ax.plot(x, test_error, label='Total Test Error', color='#d62728', linewidth=2.5)

    # Highlight the optimal complexity point
    ax.axvline(optimal_x, color='gray', linestyle=':', label='Optimal Model')

    # Add shaded regions for context
    ax.axvspan(0, optimal_x, alpha=0.1, color='blue', label='Underfitting Zone')
    ax.axvspan(optimal_x, 100, alpha=0.1, color='red', label='Overfitting Zone')

    # Interactive vertical line at current complexity
    ax.axvline(complexity, color='black', linestyle='-', alpha=0.8)

    # Find the closest index in our x array to the current slider value
    idx = (np.abs(x - complexity)).argmin()

    # Extract current values
    b2 = bias_squared[idx]
    v = variance[idx]
    tr_err = train_error[idx]
    te_err = test_error[idx]

    # Display current metrics in a text box
    textstr = (f'Current Complexity: {complexity:.1f}\n'
               f'------------------------\n'
               f'Bias²:        {b2:.1f}\n'
               f'Variance:     {v:.1f}\n'
               f'Train Error:  {tr_err:.1f}\n'
               f'Test Error:   {te_err:.1f}')
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))

    # Format the plot
    ax.set_title('The Bias-Variance Tradeoff', fontsize=14, fontweight='bold')
    ax.set_xlabel('Model Complexity', fontsize=12)
    ax.set_ylabel('Error', fontsize=12)
    ax.set_ylim(0, np.max(test_error) + 10)
    ax.set_xlim(0, 100)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.show()

# 4. Create the ipywidgets Slider and wire up interactivity.
# The slider has no description or readout so its track spans the full widget
# width, aligning as closely as possible with the plot area above.
complexity_slider = FloatSlider(
    value=50, min=0, max=100, step=0.2,
    description='',
    style={'description_width': '0px'},
    layout=Layout(width='100%', margin='0px'),
    readout=False,
    continuous_update=True
)

slider_label = HTML(
    value='<div style="text-align:center; font-weight:bold; font-size:13px;">'
          'The Bias-Variance Tradeoff</div>'
)

out = interactive_output(plot_interactive, {'complexity': complexity_slider})

display(VBox([out, complexity_slider, slider_label]))