import tkinter as tk
import config as cfg
from ui_helpers import create_rounded_rectangle


def draw_gradient(canvas_obj, width, height, top_color, bottom_color):
    r1, g1, b1 = canvas_obj.winfo_rgb(top_color)
    r2, g2, b2 = canvas_obj.winfo_rgb(bottom_color)
    steps = max(height, 1)
    for i in range(steps):
        ratio = i / steps
        r = int(r1 + (r2 - r1) * ratio)
        g = int(g1 + (g2 - g1) * ratio)
        b = int(b1 + (b2 - b1) * ratio)
        color = f"#{r:04x}{g:04x}{b:04x}"
        canvas_obj.create_line(0, i, width, i, fill=color, tags="static")


def draw_axes_and_grid(canvas_obj, model):
    plot_w, plot_h = cfg.CANVAS_W - 2 * cfg.PAD, cfg.CANVAS_H - 2 * cfg.PAD
    x0, y0 = cfg.PAD, cfg.PAD
    x1, y1 = cfg.PAD + plot_w, cfg.PAD + plot_h
    
    # Add rounded border around plot area
    create_rounded_rectangle(
        canvas_obj, x0 - 5, y0 - 5, x1 + 5, y1 + 5,
        radius=20,
        outline="#334155",
        width=2,
        fill="",
        tags="static"
    )
    
    for i in range(6):
        t = i / 5
        y = y1 - t * plot_h
        canvas_obj.create_line(x0, y, x1, y, fill=cfg.GRID_COLOR, dash=(2, 4), tags="static", capstyle=tk.ROUND)
        y_val = model.Y_MIN + t * model.Y_RANGE
        canvas_obj.create_text(x0 - 30, y, text=f"{y_val:4.1f}", fill=cfg.TEXT_COLOR, font=cfg.FONT_MAIN, tags="static")
    for i in range(6):
        t = i / 5
        x = x0 + t * plot_w
        canvas_obj.create_line(x, y0, x, y1, fill=cfg.GRID_COLOR, dash=(2, 4), tags="static", capstyle=tk.ROUND)
        x_val = model.X_MIN + t * (model.X_MAX - model.X_MIN)
        canvas_obj.create_text(x, y1 + 20, text=f"{x_val:4.1f}", fill=cfg.TEXT_COLOR, font=cfg.FONT_MAIN, tags="static")
    canvas_obj.create_line(x0, y1, x1, y1, fill=cfg.AXIS_COLOR, width=2, tags="static", capstyle=tk.ROUND)
    canvas_obj.create_line(x0, y0, x0, y1, fill=cfg.AXIS_COLOR, width=2, tags="static", capstyle=tk.ROUND)


def draw_data_points(canvas_obj, model):
    for x_val, y_val in zip(model.X, model.Y):
        px, py = model.data_to_canvas(x_val, y_val)
        canvas_obj.create_oval(px - 4, py - 4, px + 4, py + 4, fill=cfg.POINT_FILL, outline=cfg.POINT_OUTLINE, width=2, tags="points")


def init_mse_chart(mse_canvas):
    draw_gradient(mse_canvas, cfg.MSE_CANVAS_W, cfg.MSE_CANVAS_H, cfg.BG_BOTTOM, cfg.BG_TOP)
    
    # Add rounded border
    create_rounded_rectangle(
        mse_canvas, 5, 5, cfg.MSE_CANVAS_W - 5, cfg.MSE_CANVAS_H - 5,
        radius=15,
        outline="#334155",
        width=2,
        fill="",
        tags="border"
    )
    
    for i in range(5):
        y = 10 + i * (cfg.MSE_CANVAS_H - 20) / 4
        mse_canvas.create_line(10, y, cfg.MSE_CANVAS_W - 10, y, fill=cfg.GRID_COLOR, dash=(2, 4), tags="static")
    mse_canvas.create_text(14, 16, text="MSE", anchor="w", fill=cfg.TEXT_COLOR, font=cfg.FONT_BOLD, tags="static")


def update_mse_chart(mse_canvas, mse_history, current_mse):
    mse_history.append(current_mse)
    if len(mse_history) > 200:
        mse_history.pop(0)
    mse_canvas.delete("dynamic")
    if not mse_history:
        return
    max_val = max(mse_history)
    min_val = min(mse_history)
    span = max(max_val - min_val, 1e-6)
    points = []
    for idx, val in enumerate(mse_history):
        x = 10 + idx * (cfg.MSE_CANVAS_W - 20) / max(len(mse_history) - 1, 1)
        y_norm = (val - min_val) / span
        y = cfg.MSE_CANVAS_H - 20 - y_norm * (cfg.MSE_CANVAS_H - 40)
        points.append((x, y))
    for i in range(1, len(points)):
        x0, y0 = points[i - 1]
        x1, y1 = points[i]
        mse_canvas.create_line(x0, y0, x1, y1, fill=cfg.LINE_GLOW, width=2, smooth=True, tags="dynamic", capstyle=tk.ROUND, joinstyle=tk.ROUND)
    last_x, last_y = points[-1]
    mse_canvas.create_oval(last_x - 4, last_y - 4, last_x + 4, last_y + 4, fill=cfg.LINE_COLOR, outline="", tags="dynamic")
    mse_canvas.create_text(cfg.MSE_CANVAS_W - 12, last_y, text=f"{current_mse:.2f}", anchor="e", fill=cfg.TEXT_COLOR, font=cfg.FONT_BOLD, tags="dynamic")
