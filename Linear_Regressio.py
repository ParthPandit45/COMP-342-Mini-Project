import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import sklearn.metrics as metrics

# --- 1. CORE ML/MATH FUNCTIONS ---

def calculate_mse(y_true, y_pred):
    """Calculates the Mean Squared Error (MSE)."""
    # Using sklearn's function for robust calculation
    return metrics.mean_squared_error(y_true, y_pred)

def get_line_y(x, slope, intercept):
    """Calculates the y-values for the line: y = m*x + b."""
    return slope * x + intercept

# --- 2. DATA GENERATION ---

# Generate synthetic linear data with some noise
np.random.seed(42)
X = np.linspace(0, 10, 50)
TRUE_SLOPE = 1.5
TRUE_INTERCEPT = 2
Y = TRUE_SLOPE * X + TRUE_INTERCEPT + np.random.normal(0, 1.5, 50)

# --- 3. PLOTTING SETUP ---

# Initial values for the interactive line (a bad fit)
initial_slope = 0.5
initial_intercept = 10

# Create the figure and subplots
fig, ax = plt.subplots(figsize=(8, 6))
plt.subplots_adjust(left=0.1, bottom=0.35) # Make room for sliders

# Set up the main plot (The Graphics Element)
ax.scatter(X, Y, color='blue', label='Data Points', s=30)
line, = ax.plot(X, get_line_y(X, initial_slope, initial_intercept), color='red', lw=2, label='Model Line')

# Text element to display MSE (The Dynamic Data Visualization)
mse_text = ax.text(0.05, 0.95, f'MSE: {calculate_mse(Y, get_line_y(X, initial_slope, initial_intercept)):.2f}', 
                   transform=ax.transAxes, fontsize=12, verticalalignment='top')

# Scatter plot for the residuals (error lines)
residuals = ax.vlines(X, Y, get_line_y(X, initial_slope, initial_intercept), 
                      color='gray', linestyle='dashed', alpha=0.5, label='Residuals')

ax.set_title('Interactive Linear Regression and Cost Function')
ax.set_xlabel('Feature X')
ax.set_ylabel('Target Y')
ax.grid(True)
ax.legend(loc='lower right')

# --- 4. WIDGET SETUP (Interaction Elements) ---

# Define positions for the sliders
ax_slope = plt.axes([0.25, 0.2, 0.6, 0.03], facecolor='lightgray')
ax_intercept = plt.axes([0.25, 0.15, 0.6, 0.03], facecolor='lightgray')
ax_reset = plt.axes([0.75, 0.05, 0.1, 0.04])
ax_optimal = plt.axes([0.55, 0.05, 0.15, 0.04])

# Create the sliders
slope_slider = Slider(
    ax_slope, 'Slope (m)', 
    valmin=-3, valmax=5, valinit=initial_slope, valstep=0.1
)
intercept_slider = Slider(
    ax_intercept, 'Intercept (b)',
    valmin=-10, valmax=20, valinit=initial_intercept, valstep=0.5
)

reset_button = Button(ax_reset, 'Reset', color='lightskyblue', hovercolor='0.975')
optimal_button = Button(ax_optimal, 'Show Optimal', color='lightcoral', hovercolor='0.975')

# --- 5. UPDATE AND CALLBACK LOGIC ---

def update(val):
    """
    Function called when a slider value changes.
    This drives the dynamic graphics update.
    """
    # 1. FIX: Declare global variable FIRST
    global residuals 
    
    new_slope = slope_slider.val
    new_intercept = intercept_slider.val
    
    # 2. Update the line (Graphics)
    Y_pred = get_line_y(X, new_slope, new_intercept)
    line.set_ydata(Y_pred)
    
    # 3. Update the residuals (Error Visualization)
    # Now, residuals.remove() is valid because residuals is declared global
    residuals.remove() 
    residuals = ax.vlines(X, Y, Y_pred, color='gray', linestyle='dashed', alpha=0.5)

    # 4. Update the MSE Text (Cost Function Visualization)
    current_mse = calculate_mse(Y, Y_pred)
    mse_text.set_text(f'MSE: {current_mse:.2f}')
    
    # Redraw the canvas
    fig.canvas.draw_idle()
    
# Connect the update function to the slider events
slope_slider.on_changed(update)
intercept_slider.on_changed(update)

def reset(event):
    """Resets the sliders to initial values."""
    slope_slider.reset()
    intercept_slider.reset()

def show_optimal(event):
    """Sets the sliders to the known optimal values."""
    slope_slider.set_val(TRUE_SLOPE)
    intercept_slider.set_val(TRUE_INTERCEPT)

# Connect the buttons to their respective functions
reset_button.on_clicked(reset)
optimal_button.on_clicked(show_optimal)

# --- 6. DISPLAY ---

# Initial call to update to ensure everything is drawn correctly at startup
update(None) 
plt.show()