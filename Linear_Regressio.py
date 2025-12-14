import tkinter as tk
import numpy as np
import sklearn.metrics as metrics

# --- 1. CONFIGURATION & STATE ---
CANVAS_W, CANVAS_H, PAD = 700, 500, 50
LEARNING_RATE = 0.01  # Step size (affects convergence)
TRUE_S, TRUE_I = 1.5, 2.0
INITIAL_S, INITIAL_I = 0.0, 5.0
DELAY = 20  # REDUCED DELAY for faster animation (was 100)
MAX_ITER = 500
N_POINTS = 50

# Global State
current_slope, current_intercept = INITIAL_S, INITIAL_I
iteration_count = 0
animation_job = None
animation_running = False
line_id, residuals_ids = None, [] # Tkinter IDs

# Generate Data
np.random.seed(42)
X = np.linspace(0, 10, N_POINTS)
Y = TRUE_S * X + TRUE_I + np.random.normal(0, 1.5, N_POINTS)
N = len(X)
X_MAX, X_MIN, Y_MAX, Y_MIN = X.max(), X.min(), Y.max(), Y.min()
Y_RANGE = Y_MAX - Y_MIN

# --- 2. HELPERS ---
def data_to_canvas(x_data, y_data):
    plot_w, plot_h = CANVAS_W - 2 * PAD, CANVAS_H - 2 * PAD
    x_pixel = PAD + (x_data - X_MIN) * (plot_w / (X_MAX - X_MIN))
    y_norm = (y_data - Y_MIN) / Y_RANGE
    y_pixel = CANVAS_H - PAD - y_norm * plot_h
    return x_pixel, y_pixel

# --- 3. CORE LOGIC (Gradient Descent & Visualization) ---
def gradient_descent_step():
    global current_slope, current_intercept, iteration_count, animation_job
    
    if not animation_running or iteration_count >= MAX_ITER:
        iter_label.config(text=f"Training Complete ({iteration_count} Iterations)")
        return

    # Calculate Gradients
    Y_pred = current_slope * X + current_intercept
    error = Y_pred - Y
    d_intercept = (2/N) * np.sum(error)
    d_slope = (2/N) * np.sum(error * X)

    # Update Parameters
    current_intercept -= LEARNING_RATE * d_intercept
    current_slope -= LEARNING_RATE * d_slope
    
    update_visualization(Y_pred)
    iteration_count += 1
    
    animation_job = root.after(DELAY, gradient_descent_step)

def update_visualization(Y_pred):
    global line_id, residuals_ids
    
    current_mse = metrics.mean_squared_error(Y, Y_pred)

    # 1. Update Line
    if line_id: canvas.delete(line_id)
    y1, y2 = current_slope * X_MIN + current_intercept, current_slope * X_MAX + current_intercept
    p1x, p1y = data_to_canvas(X_MIN, y1); p2x, p2y = data_to_canvas(X_MAX, y2)
    line_id = canvas.create_line(p1x, p1y, p2x, p2y, fill='red', width=3)

    # 2. Update Residuals
    for res_id in residuals_ids: canvas.delete(res_id)
    residuals_ids.clear()
    for x_data, y_actual, y_pred in zip(X, Y, Y_pred):
        px, py_actual = data_to_canvas(x_data, y_actual)
        _, py_pred = data_to_canvas(x_data, y_pred)
        residuals_ids.append(canvas.create_line(px, py_actual, px, py_pred, fill='gray', dash=(4, 2)))

    # 3. Update Labels
    mse_label.config(text=f"MSE: {current_mse:.2f}")
    iter_label.config(text=f"Iteration: {iteration_count}")
    canvas.tag_raise(line_id)

# --- 4. CONTROL FUNCTIONS (Consolidated) ---
def handle_control(action):
    global current_slope, current_intercept, iteration_count, animation_running, animation_job
    
    if action == 'start' and not animation_running:
        animation_running = True
        gradient_descent_step()
    
    elif action == 'pause':
        animation_running = False
        if animation_job: root.after_cancel(animation_job)
        iter_label.config(text=f"Paused at Iteration {iteration_count}")
    
    elif action == 'reset':
        handle_control('pause')
        current_slope, current_intercept = INITIAL_S, INITIAL_I
        iteration_count = 0
        
        # Calculate Y_pred for the initial state update
        Y_pred = current_slope * X + current_intercept
        update_visualization(Y_pred)
        iter_label.config(text="Ready to Start Training")


# --- 5. TKINTER SETUP & EXECUTION ---
root = tk.Tk()
root.title("Fast GD Animation")
canvas = tk.Canvas(root, width=CANVAS_W, height=CANVAS_H, bg="white")
canvas.pack(pady=10)

# Labels
mse_label = tk.Label(root, font=('Arial', 14, 'bold')); mse_label.pack()
iter_label = tk.Label(root, font=('Arial', 12)); iter_label.pack()

# Draw static data points
for x, y in zip(X, Y):
    px, py = data_to_canvas(x, y)
    canvas.create_oval(px - 3, py - 3, px + 3, py + 3, fill='blue', outline='blue')

# Buttons
control_frame = tk.Frame(root); control_frame.pack(fill='x', pady=10)
tk.Button(control_frame, text="Start Training", command=lambda: handle_control('start')).pack(side=tk.LEFT, padx=10)
tk.Button(control_frame, text="Pause Training", command=lambda: handle_control('pause')).pack(side=tk.LEFT, padx=10)
tk.Button(control_frame, text="Reset Model", command=lambda: handle_control('reset')).pack(side=tk.LEFT, padx=10)

# --- 6. EXECUTION ---
handle_control('reset') # Initialize the display
root.mainloop()