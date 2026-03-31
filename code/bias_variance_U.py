import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

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

# 3. Setup the figure and main axis
fig, ax = plt.subplots(figsize=(10, 6))
plt.subplots_adjust(bottom=0.25) # Leave room for the slider

# Plot the static curves
ax.plot(x, bias_squared, label='Bias²', color='#1f77b4', linestyle='--')
ax.plot(x, variance, label='Variance', color='#2ca02c', linestyle='--')
ax.plot(x, train_error, label='Training Error', color='#ff7f0e')
ax.plot(x, test_error, label='Total Test Error', color='#d62728', linewidth=2.5)

# Highlight the optimal complexity point (minimum test error)
optimal_idx = np.argmin(test_error)
optimal_x = x[optimal_idx]
ax.axvline(optimal_x, color='gray', linestyle=':', label='Optimal Model')

# Add shaded regions for context
ax.axvspan(0, optimal_x, alpha=0.1, color='blue', label='Underfitting Zone')
ax.axvspan(optimal_x, 100, alpha=0.1, color='red', label='Overfitting Zone')

# Setup the interactive vertical line and text box
initial_complexity = 50
vline = ax.axvline(initial_complexity, color='black', linestyle='-', alpha=0.8)
text_box = ax.text(0.05, 0.95, '', transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))

# Format the plot
ax.set_title('The Bias-Variance Tradeoff', fontsize=14, fontweight='bold')
ax.set_xlabel('Model Complexity', fontsize=12)
ax.set_ylabel('Error', fontsize=12)
ax.set_ylim(0, np.max(test_error) + 10)
ax.set_xlim(0, 100)
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

# 4. Create the Slider
ax_slider = plt.axes([0.15, 0.1, 0.7, 0.03]) # [left, bottom, width, height]
complexity_slider = Slider(
    ax=ax_slider,
    label='Complexity',
    valmin=0,
    valmax=100,
    valinit=initial_complexity,
    color='#7f7f7f'
)

# 5. Define the update function for interactivity
def update(val):
    current_x = complexity_slider.val

    # Move the vertical line
    vline.set_xdata([current_x, current_x])

    # Find the closest index in our x array to the current slider value
    idx = (np.abs(x - current_x)).argmin()

    # Extract current values
    b2 = bias_squared[idx]
    v = variance[idx]
    tr_err = train_error[idx]
    te_err = test_error[idx]

    # Update the text box with current metrics
    textstr = (f'Current Complexity: {current_x:.1f}\n'
               f'------------------------\n'
               f'Bias²:        {b2:.1f}\n'
               f'Variance:     {v:.1f}\n'
               f'Train Error:  {tr_err:.1f}\n'
               f'Test Error:   {te_err:.1f}')
    text_box.set_text(textstr)

    # Redraw the canvas
    fig.canvas.draw_idle()

# Initialize the text box with the starting values
update(initial_complexity)

# Connect the slider movement to the update function
complexity_slider.on_changed(update)

# Show the interactive plot
plt.show()