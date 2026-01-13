import tkinter as tk
import tkinter.font as tkfont
import numpy as np
import sklearn.metrics as metrics
import config as cfg
from data_model import DataModel
import visuals as vis
from metrics import MetricsTracker
from optimizers import SGD, Momentum, Adam
from exporter import DataExporter
from data_generator import DataGenerator
from educational import EducationalPanel, CanvasAnnotations
from animations import AnimationEngine, VisualEffects

# --- 1. STATE ---
current_slope, current_intercept = cfg.INITIAL_S, cfg.INITIAL_I
iteration_count = 0
animation_job = None
animation_running = False
line_id, residuals_ids = None, []
mse_history = []
learning_rate_var = None
speed_var = None
residuals_var = None
optimizer_var = None
dataset_var = None
model = DataModel(seed=42)
metrics_tracker = MetricsTracker()
current_optimizer = SGD(cfg.LEARNING_RATE)
edu_panel = None
anim_engine = None
prev_mse = None
show_educational = True

# --- 2. HELPERS ---

def set_status(text):
    status_label.config(text=text)


def update_metric_labels(current_mse):
    mse_label.config(text=f"MSE: {current_mse:.2f}")
    iter_label.config(text=f"Iteration: {iteration_count}")
    slope_label.config(text=f"Slope: {current_slope:.3f}")
    intercept_label.config(text=f"Intercept: {current_intercept:.3f}")


def clear_dynamic_layers():
    global line_id, residuals_ids
    if line_id:
        canvas.delete(line_id)
    canvas.delete("line-glow")
    for res_id in residuals_ids:
        canvas.delete(res_id)
    line_id = None
    residuals_ids = []


def refresh_visual_state():
    Y_pred = current_slope * model.X + current_intercept
    update_visualization(Y_pred)


def regenerate_data(seed=None):
    model.generate(seed=seed)
    canvas.delete("static")
    vis.draw_gradient(canvas, cfg.CANVAS_W, cfg.CANVAS_H, cfg.BG_TOP, cfg.BG_BOTTOM)
    vis.draw_axes_and_grid(canvas, model)
    canvas.delete("points")
    vis.draw_data_points(canvas, model)
    clear_dynamic_layers()
    mse_history.clear()
    metrics_tracker.clear()
    mse_canvas.delete("dynamic")
    
    # Redraw educational elements
    if show_educational:
        VisualEffects.draw_best_fit_comparison(canvas, model, current_slope, current_intercept, cfg.TRUE_S, cfg.TRUE_I)
    
    set_status("New data")


def load_dataset(dataset_type):
    """Load a different dataset type."""
    if dataset_type == "polynomial":
        model.X, model.Y = DataGenerator.polynomial(50, [0.5, 0.3, 0.02])
    elif dataset_type == "sinusoidal":
        model.X, model.Y = DataGenerator.sinusoidal(50, amplitude=2, frequency=0.5)
    elif dataset_type == "exponential":
        model.X, model.Y = DataGenerator.exponential(50)
    else:  # linear
        model.X, model.Y = DataGenerator.noisy_linear(50, noise_level=0.75)
    
    model.N = len(model.X)
    model.X_MAX, model.X_MIN = model.X.max(), model.X.min()
    model.Y_MAX, model.Y_MIN = model.Y.max(), model.Y.min()
    model.Y_RANGE = max(model.Y_MAX - model.Y_MIN, 1e-6)
    
    canvas.delete("static")
    vis.draw_gradient(canvas, cfg.CANVAS_W, cfg.CANVAS_H, cfg.BG_TOP, cfg.BG_BOTTOM)
    vis.draw_axes_and_grid(canvas, model)
    canvas.delete("points")
    vis.draw_data_points(canvas, model)
    clear_dynamic_layers()
    mse_history.clear()
    metrics_tracker.clear()
    mse_canvas.delete("dynamic")
    set_status(f"Loaded {dataset_type}")


def switch_optimizer(opt_name):
    """Switch to a different optimizer."""
    global current_optimizer
    lr = cfg.LEARNING_RATE  # Use fixed learning rate from config
    if opt_name == "SGD":
        current_optimizer = SGD(lr)
    elif opt_name == "Momentum":
        current_optimizer = Momentum(lr, momentum=0.9)
    elif opt_name == "Adam":
        current_optimizer = Adam(lr)
    set_status(f"Optimizer: {opt_name}")


def export_training_state():
    """Export current training state and metrics."""
    state = {
        "slope": float(current_slope),
        "intercept": float(current_intercept),
        "iteration": iteration_count,
        "optimizer": optimizer_var.get() if optimizer_var else "SGD",
        "metrics": metrics_tracker.summary()
    }
    filename = DataExporter.export_model_state(state)
    set_status(f"Exported to {filename}")


# --- 3. CORE LOGIC (Gradient Descent & Visualization) ---
def run_one_iteration():
    global current_slope, current_intercept, iteration_count, prev_mse

    if iteration_count >= cfg.MAX_ITER:
        iter_label.config(text=f"Training Complete ({iteration_count} Iterations)")
        set_status("Complete")
        return False

    old_slope, old_intercept = current_slope, current_intercept
    Y_pred = current_slope * model.X + current_intercept
    error = Y_pred - model.Y
    lr = cfg.LEARNING_RATE  # Use fixed learning rate from config
    d_intercept = (2 / model.N) * np.sum(error)
    d_slope = (2 / model.N) * np.sum(error * model.X)

    # Use selected optimizer
    current_optimizer.learning_rate = lr
    current_slope, current_intercept = current_optimizer.step(current_slope, current_intercept, d_slope, d_intercept)

    # Visualize gradient step
    if show_educational and anim_engine:
        VisualEffects.create_gradient_arrow(canvas, model, old_slope, old_intercept, current_slope, current_intercept)

    update_visualization(Y_pred)
    iteration_count += 1
    return True


def gradient_descent_step():
    global animation_job, animation_running

    if not animation_running:
        return
    if not run_one_iteration():
        animation_running = False
        return
    delay = max(1, int(speed_var.get()))
    animation_job = root.after(delay, gradient_descent_step)


def update_visualization(Y_pred):
    global line_id, residuals_ids, prev_mse

    current_mse = metrics.mean_squared_error(model.Y, Y_pred)
    metrics_tracker.update(model.Y, Y_pred, iteration_count)
    
    # Show error contributions
    if show_educational and anim_engine and iteration_count % 10 == 0:
        anim_engine.clear_highlights()
        points = [(model.data_to_canvas(x, y)) for x, y in zip(model.X, model.Y)]
        errors = Y_pred - model.Y
        anim_engine.show_error_contribution(points, errors)

    canvas.delete("line-glow")
    if line_id:
        canvas.delete(line_id)
    y1 = current_slope * model.X_MIN + current_intercept
    y2 = current_slope * model.X_MAX + current_intercept
    p1x, p1y = model.data_to_canvas(model.X_MIN, y1)
    p2x, p2y = model.data_to_canvas(model.X_MAX, y2)
    line_id = canvas.create_line(p1x, p1y, p2x, p2y, fill=cfg.LINE_COLOR, width=4, tags="line", capstyle=tk.ROUND, joinstyle=tk.ROUND, smooth=True)
    canvas.create_line(p1x, p1y, p2x, p2y, fill=cfg.LINE_GLOW, width=8, stipple="gray50", tags="line-glow", capstyle=tk.ROUND, joinstyle=tk.ROUND, smooth=True)

    for res_id in residuals_ids:
        canvas.delete(res_id)
    residuals_ids.clear()
    if residuals_var.get():
        for x_data, y_actual, y_pred in zip(model.X, model.Y, Y_pred):
            px, py_actual = model.data_to_canvas(x_data, y_actual)
            _, py_pred = model.data_to_canvas(x_data, y_pred)
            residuals_ids.append(canvas.create_line(px, py_actual, px, py_pred, fill=cfg.RESIDUAL_COLOR, dash=(4, 2), width=1, tags="residual", capstyle=tk.ROUND))

    update_metric_labels(current_mse)
    vis.update_mse_chart(mse_canvas, mse_history, current_mse)
    canvas.tag_raise("points")
    canvas.tag_raise("line")
    canvas.tag_raise("line-glow")


# --- 4A. UI WIDGETS: Rounded Button ---
class RoundedButton(tk.Canvas):
    def __init__(self, master, text, command=None, width=None, height=None, radius=14,
                 bg="#0ea5e9", fg="white", hover_bg=None, active_bg=None,
                 border_color="#0b6fa1", border_width=1,
                 hover_border_color="#38bdf8",
                 shadow=True, shadow_color="#000000", shadow_offset=(0, 2), shadow_layers=2,
                 font=("Helvetica", 11, "bold"), pill=False,
                 padding_x=14, padding_y=8, cursor="hand2", **kwargs):
        # Measure text if width/height not provided
        self._font = tkfont.Font(font=font)
        text_w = self._font.measure(text)
        text_h = self._font.metrics("linespace")
        self._base_radius = int(radius)
        self._text = text
        self._fg = fg
        self._bg = bg
        self._hover_bg = hover_bg or bg
        self._active_bg = active_bg or self._hover_bg
        self._border_color = border_color
        self._hover_border_color = hover_border_color or border_color
        self._border_width = max(0, int(border_width))
        self._shadow = shadow
        self._shadow_color = shadow_color
        self._shadow_offset = tuple(shadow_offset)
        self._shadow_layers = max(0, int(shadow_layers))
        self._command = command
        self._pressed = False
        self._hover = False
        self._pill = bool(pill)
        w = width or (text_w + 2 * padding_x)
        h = height or (text_h + 2 * padding_y)
        super().__init__(master, width=w, height=h, highlightthickness=0, bg=master.cget("bg"), bd=0, **kwargs)
        self.configure(cursor=cursor)

        self._draw(self._bg)

        # Event bindings
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<ButtonPress-1>", self._on_press)
        self.bind("<ButtonRelease-1>", self._on_release)

    def _calc_radius(self, w, h):
        return min((h // 2) if self._pill else self._base_radius, w // 2, h // 2)

    def _draw(self, fill_color):
        self.delete("all")
        w = int(self.cget("width"))
        h = int(self.cget("height"))
        r = self._calc_radius(w, h)
        inset = 1

        # Shadow layers (offset down-right)
        if self._shadow and not self._pressed:
            ox, oy = self._shadow_offset
            for i in range(self._shadow_layers):
                alpha_stipple = "gray50" if i == 0 else "gray25"
                sx1, sy1 = inset + ox + i, inset + oy + i
                sx2, sy2 = w - inset + ox + i, h - inset + oy + i
                self._draw_round_rect(sx1, sy1, sx2, sy2, r, fill=self._shadow_color, stipple=alpha_stipple)

        # Border (outer layer with color)
        if self._border_width > 0:
            border_col = self._hover_border_color if self._hover else self._border_color
            self._draw_round_rect(inset, inset, w - inset, h - inset, r, fill=border_col)
            # Inner fill (inset by border width)
            inner_inset = inset + self._border_width
            inner_r = max(0, r - self._border_width)
            self._draw_round_rect(inner_inset, inner_inset, w - inner_inset, h - inner_inset, inner_r, fill=fill_color)
        else:
            self._draw_round_rect(inset, inset, w - inset, h - inset, r, fill=fill_color)

        # Text (pressed state nudges down 1px)
        tx, ty = w // 2, h // 2 + (1 if self._pressed else 0)
        self.create_text(tx, ty, text=self._text, fill=self._fg, font=self._font)

    def _draw_round_rect(self, x1, y1, x2, y2, r, fill, stipple=None):
        """Draw a rounded rectangle with smooth corners using ovals and rectangles."""
        r = max(0, min(r, (x2 - x1) // 2, (y2 - y1) // 2))
        # Central rectangles (avoiding corner areas)
        self.create_rectangle(x1 + r, y1, x2 - r, y2, fill=fill, outline="", stipple=stipple)
        self.create_rectangle(x1, y1 + r, x2, y2 - r, fill=fill, outline="", stipple=stipple)
        # Four corners as quarter-ovals
        self.create_oval(x1, y1, x1 + 2 * r, y1 + 2 * r, fill=fill, outline="", stipple=stipple)
        self.create_oval(x2 - 2 * r, y1, x2, y1 + 2 * r, fill=fill, outline="", stipple=stipple)
        self.create_oval(x1, y2 - 2 * r, x1 + 2 * r, y2, fill=fill, outline="", stipple=stipple)
        self.create_oval(x2 - 2 * r, y2 - 2 * r, x2, y2, fill=fill, outline="", stipple=stipple)

    def _inside(self, x, y):
        """Check if point (x, y) is inside the button canvas."""
        w = int(self.cget("width"))
        h = int(self.cget("height"))
        return 0 <= x <= w and 0 <= y <= h

    def _on_enter(self, _):
        """Handle mouse enter."""
        self._hover = True
        if not self._pressed:
            self._draw(self._hover_bg)

    def _on_leave(self, _):
        """Handle mouse leave."""
        self._hover = False
        if not self._pressed:
            self._draw(self._bg)

    def _on_press(self, _):
        """Handle mouse press."""
        self._pressed = True
        self._draw(self._active_bg)

    def _on_release(self, event):
        """Handle mouse release."""
        inside = self._inside(event.x, event.y)
        self._pressed = False
        self._draw(self._hover_bg if (inside and self._hover) else self._bg)
        if inside and callable(self._command):
            self._command()


# --- 4. CONTROL FUNCTIONS (Consolidated) ---
def handle_control(action):
    global current_slope, current_intercept, iteration_count, animation_running, animation_job

    if action == "start" and not animation_running:
        animation_running = True
        set_status("Running")
        gradient_descent_step()

    elif action == "pause":
        animation_running = False
        if animation_job:
            root.after_cancel(animation_job)
        iter_label.config(text=f"Paused at Iteration {iteration_count}")
        set_status("Paused")

    elif action == "step":
        animation_running = False
        if animation_job:
            root.after_cancel(animation_job)
        set_status("Stepping")
        run_one_iteration()

    elif action == "reset":
        handle_control("pause")
        current_slope, current_intercept = cfg.INITIAL_S, cfg.INITIAL_I
        iteration_count = 0
        mse_history.clear()
        metrics_tracker.clear()
        clear_dynamic_layers()
        mse_canvas.delete("dynamic")
        Y_pred = current_slope * model.X + current_intercept
        update_visualization(Y_pred)
        iter_label.config(text="Ready to Start Training")
        set_status("Idle")

    elif action == "regen":
        handle_control("pause")
        regenerate_data()
        current_slope, current_intercept = cfg.INITIAL_S, cfg.INITIAL_I
        iteration_count = 0
        Y_pred = current_slope * model.X + current_intercept
        update_visualization(Y_pred)
        iter_label.config(text="New data loaded")


# --- 5. TKINTER SETUP & EXECUTION ---
root = tk.Tk()
root.title("Gradient Descent Visualizer")
root.configure(bg=cfg.BG_BOTTOM)

learning_rate_var = tk.DoubleVar(root, value=cfg.LEARNING_RATE)
speed_var = tk.IntVar(root, value=cfg.DELAY)
residuals_var = tk.BooleanVar(root, value=True)
optimizer_var = tk.StringVar(root, value="SGD")
dataset_var = tk.StringVar(root, value="linear")

# Main container with side panels
main_container = tk.Frame(root, bg=cfg.BG_BOTTOM)
main_container.pack(fill="both", expand=True, padx=10, pady=10)

# Left sidebar for controls
left_panel = tk.Frame(main_container, bg=cfg.BG_BOTTOM, width=200)
left_panel.pack(side=tk.LEFT, fill="y", padx=(0, 10))

# Center panel for canvases
center_panel = tk.Frame(main_container, bg=cfg.BG_BOTTOM)
center_panel.pack(side=tk.LEFT, fill="both", expand=True)

canvas = tk.Canvas(center_panel, width=cfg.CANVAS_W, height=cfg.CANVAS_H, highlightthickness=0)
canvas.pack()
mse_canvas = tk.Canvas(center_panel, width=cfg.MSE_CANVAS_W, height=cfg.MSE_CANVAS_H, highlightthickness=0)
mse_canvas.pack(pady=(10, 0))

# Right sidebar for additional controls
right_panel = tk.Frame(main_container, bg=cfg.BG_BOTTOM, width=200)
right_panel.pack(side=tk.LEFT, fill="y", padx=(10, 0))

vis.draw_gradient(canvas, cfg.CANVAS_W, cfg.CANVAS_H, cfg.BG_TOP, cfg.BG_BOTTOM)
vis.draw_axes_and_grid(canvas, model)
vis.draw_data_points(canvas, model)
vis.init_mse_chart(mse_canvas)

# Initialize animation engine
anim_engine = AnimationEngine(canvas)
VisualEffects.draw_best_fit_comparison(canvas, model, current_slope, current_intercept, cfg.TRUE_S, cfg.TRUE_I)

# --- LEFT PANEL: Training Controls ---
tk.Label(left_panel, text="Training Controls", font=cfg.FONT_BOLD, fg=cfg.TEXT_COLOR, bg=cfg.BG_BOTTOM).pack(pady=(0, 15))

# Delay Slider (Vertical)
tk.Label(left_panel, text="Delay (ms)", font=cfg.FONT_MAIN, fg=cfg.TEXT_COLOR, bg=cfg.BG_BOTTOM).pack(pady=(10, 5))
delay_scale = tk.Scale(left_panel, from_=200, to=1, resolution=1, variable=speed_var,
                       orient=tk.VERTICAL, length=250, highlightthickness=0, bg=cfg.BG_BOTTOM,
                       fg=cfg.TEXT_COLOR, troughcolor="#0b1220", sliderrelief=tk.FLAT)
delay_scale.pack(pady=(0, 10))

# Show Residuals Checkbox
tk.Checkbutton(left_panel, text="Show Residuals", variable=residuals_var, onvalue=True, offvalue=False,
               bg=cfg.BG_BOTTOM, fg=cfg.TEXT_COLOR, activebackground=cfg.BG_BOTTOM, 
               activeforeground=cfg.TEXT_COLOR, selectcolor=cfg.BG_BOTTOM, 
               font=cfg.FONT_MAIN, command=refresh_visual_state).pack(pady=15)

# Educational Features Toggle
tk.Label(left_panel, text="Visual Aids", font=cfg.FONT_BOLD, fg=cfg.TEXT_COLOR, bg=cfg.BG_BOTTOM).pack(pady=(20, 10))

def toggle_educational():
    global show_educational
    show_educational = not show_educational
    if show_educational:
        edu_panel.show_all()
        VisualEffects.draw_best_fit_comparison(canvas, model, current_slope, current_intercept, cfg.TRUE_S, cfg.TRUE_I)
    else:
        edu_panel.hide_all()
        canvas.delete("true_line")
        anim_engine.clear_highlights()

RoundedButton(left_panel, text="Toggle Formulas", command=toggle_educational,
              font=("Helvetica", 9), bg="#8b5cf6", hover_bg="#7c3aed", active_bg="#6d28d9",
              border_color="#6d28d9", hover_border_color="#8b5cf6",
              radius=2, pill=False, padding_x=14, padding_y=8, shadow=True, shadow_layers=2).pack(pady=5)

# --- RIGHT PANEL: Educational Content & Settings ---
tk.Label(right_panel, text="Algorithm Reference", font=cfg.FONT_BOLD, fg=cfg.TEXT_COLOR, bg=cfg.BG_BOTTOM).pack(pady=(0, 10))

# Create educational panel
edu_panel = EducationalPanel(right_panel)
edu_panel.create_formula_panel()
edu_panel.create_legend_panel()

tk.Label(right_panel, text="Model Settings", font=cfg.FONT_BOLD, fg=cfg.TEXT_COLOR, bg=cfg.BG_BOTTOM).pack(pady=(15, 10))

# Optimizer Selection
tk.Label(right_panel, text="Optimizer", font=cfg.FONT_MAIN, fg=cfg.TEXT_COLOR, bg=cfg.BG_BOTTOM).pack(pady=(10, 5))
optimizer_menu = tk.OptionMenu(right_panel, optimizer_var, "SGD", "Momentum", "Adam", command=switch_optimizer)
optimizer_menu.config(bg="#0ea5e9", fg="white", highlightthickness=0, width=15)
optimizer_menu.pack(pady=(0, 15))

# Dataset Selection
tk.Label(right_panel, text="Dataset Type", font=cfg.FONT_MAIN, fg=cfg.TEXT_COLOR, bg=cfg.BG_BOTTOM).pack(pady=(10, 5))
dataset_menu = tk.OptionMenu(right_panel, dataset_var, "linear", "polynomial", "sinusoidal", "exponential", 
                            command=lambda x: load_dataset(x))
dataset_menu.config(bg="#0ea5e9", fg="white", highlightthickness=0, width=15)
dataset_menu.pack(pady=(0, 15))

# Export Button
RoundedButton(right_panel, text="Export State", command=export_training_state,
              font=("Helvetica", 10, "bold"), bg="#10b981", hover_bg="#059669", active_bg="#047857",
              border_color="#065f46", hover_border_color="#34d399",
              radius=2, pill=False, padding_x=14, padding_y=9, shadow=True, shadow_layers=2).pack(pady=10)

# --- BOTTOM: Info and Controls ---
info_frame = tk.Frame(root, bg=cfg.BG_BOTTOM)
info_frame.pack(pady=6)

mse_label = tk.Label(info_frame, font=cfg.FONT_TITLE, fg=cfg.TEXT_COLOR, bg=cfg.BG_BOTTOM)
iter_label = tk.Label(info_frame, font=cfg.FONT_MAIN, fg=cfg.TEXT_COLOR, bg=cfg.BG_BOTTOM)
slope_label = tk.Label(info_frame, font=cfg.FONT_MAIN, fg=cfg.TEXT_COLOR, bg=cfg.BG_BOTTOM)
intercept_label = tk.Label(info_frame, font=cfg.FONT_MAIN, fg=cfg.TEXT_COLOR, bg=cfg.BG_BOTTOM)
status_label = tk.Label(info_frame, font=cfg.FONT_BOLD, fg=cfg.LINE_COLOR, bg=cfg.BG_BOTTOM)

mse_label.grid(row=0, column=0, padx=12)
iter_label.grid(row=0, column=1, padx=12)
slope_label.grid(row=0, column=2, padx=12)
intercept_label.grid(row=0, column=3, padx=12)
status_label.grid(row=0, column=4, padx=12)

control_frame = tk.Frame(root, bg=cfg.BG_BOTTOM)
control_frame.pack(fill="x", pady=10)

RoundedButton(control_frame, text="Start Training", command=lambda: handle_control("start"),
              font=("Helvetica", 11, "bold"), bg="#0ea5e9", hover_bg="#0284c7", active_bg="#0369a1",
              border_color="#075985", hover_border_color="#38bdf8",
              radius=2, pill=False, padding_x=18, padding_y=10, shadow=True, shadow_layers=2).pack(side=tk.LEFT, padx=10)
RoundedButton(control_frame, text="Pause Training", command=lambda: handle_control("pause"),
              font=("Helvetica", 11, "bold"), bg="#0ea5e9", hover_bg="#0284c7", active_bg="#0369a1",
              border_color="#075985", hover_border_color="#38bdf8",
              radius=2, pill=False, padding_x=18, padding_y=10, shadow=True, shadow_layers=2).pack(side=tk.LEFT, padx=10)
RoundedButton(control_frame, text="Step Once", command=lambda: handle_control("step"),
              font=("Helvetica", 11, "bold"), bg="#0ea5e9", hover_bg="#0284c7", active_bg="#0369a1",
              border_color="#075985", hover_border_color="#38bdf8",
              radius=2, pill=False, padding_x=18, padding_y=10, shadow=True, shadow_layers=2).pack(side=tk.LEFT, padx=10)
RoundedButton(control_frame, text="Reset Model", command=lambda: handle_control("reset"),
              font=("Helvetica", 11, "bold"), bg="#0ea5e9", hover_bg="#0284c7", active_bg="#0369a1",
              border_color="#075985", hover_border_color="#38bdf8",
              radius=2, pill=False, padding_x=18, padding_y=10, shadow=True, shadow_layers=2).pack(side=tk.LEFT, padx=10)
RoundedButton(control_frame, text="Regenerate Data", command=lambda: handle_control("regen"),
              font=("Helvetica", 11, "bold"), bg="#0ea5e9", hover_bg="#0284c7", active_bg="#0369a1",
              border_color="#075985", hover_border_color="#38bdf8",
              radius=2, pill=False, padding_x=18, padding_y=10, shadow=True, shadow_layers=2).pack(side=tk.LEFT, padx=10)

# --- 6. EXECUTION ---
handle_control("reset")
root.mainloop()