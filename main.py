"""OpenGL-only entrypoint.

Run this file to start the OpenGL UI.
"""

from __future__ import annotations

import numpy as np
import pygame
from OpenGL.GL import (
    GL_BLEND,
    GL_COLOR_BUFFER_BIT,
    GL_LINE_SMOOTH,
    GL_LINES,
    GL_LINEAR,
    GL_MODELVIEW,
    GL_MULTISAMPLE,
    GL_ONE_MINUS_SRC_ALPHA,
    GL_POINTS,
    GL_POINT_SMOOTH,
    GL_PROJECTION,
    GL_RGBA,
    GL_SRC_ALPHA,
    GL_TEXTURE_2D,
    GL_TEXTURE_MAG_FILTER,
    GL_TEXTURE_MIN_FILTER,
    GL_TRIANGLES,
    GL_UNSIGNED_BYTE,
    glBegin,
    glBindTexture,
    glBlendFunc,
    glClear,
    glClearColor,
    glColor4f,
    glDisable,
    glEnable,
    glEnd,
    glGenTextures,
    glLineWidth,
    glLoadIdentity,
    glMatrixMode,
    glOrtho,
    glPointSize,
    glTexCoord2f,
    glTexImage2D,
    glTexParameteri,
    glTexSubImage2D,
    glVertex2f,
    glViewport,
)

from algorithms.metrics import MetricsTracker
from algorithms.optimizers import Adam, Momentum, SGD
from core import config as cfg
from model.data_model import DataModel
from ui.opengl_ui import Layout, UIManager


def hex_to_rgb01(hex_color: str) -> tuple[float, float, float]:
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16) / 255.0
    g = int(hex_color[2:4], 16) / 255.0
    b = int(hex_color[4:6], 16) / 255.0
    return r, g, b


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    err = y_pred - y_true
    return float(np.mean(err * err))


def compute_step(model: DataModel, slope: float, intercept: float, optimizer) -> tuple[float, float, float]:
    """One gradient descent step; returns (new_slope, new_intercept, current_mse)."""
    y_pred = slope * model.X + intercept
    error = y_pred - model.Y

    d_intercept = (2.0 / model.N) * float(np.sum(error))
    d_slope = (2.0 / model.N) * float(np.sum(error * model.X))

    optimizer.learning_rate = cfg.LEARNING_RATE
    new_slope, new_intercept = optimizer.step(slope, intercept, d_slope, d_intercept)
    current_mse = mse(model.Y, y_pred)
    return float(new_slope), float(new_intercept), current_mse


def draw_gradient_background(w: int, h: int, top_hex: str, bottom_hex: str) -> None:
    # Draw in screen-space using an orthographic projection (pixel coordinates).
    # Two triangles with per-vertex colors for a smooth gradient (much faster than per-scanline lines).
    top = hex_to_rgb01(top_hex)
    bottom = hex_to_rgb01(bottom_hex)

    set_screen_projection(w, h)

    glBegin(GL_TRIANGLES)
    # triangle 1
    glColor4f(bottom[0], bottom[1], bottom[2], 1.0)
    glVertex2f(0, 0)
    glColor4f(bottom[0], bottom[1], bottom[2], 1.0)
    glVertex2f(w, 0)
    glColor4f(top[0], top[1], top[2], 1.0)
    glVertex2f(w, h)
    # triangle 2
    glColor4f(bottom[0], bottom[1], bottom[2], 1.0)
    glVertex2f(0, 0)
    glColor4f(top[0], top[1], top[2], 1.0)
    glVertex2f(w, h)
    glColor4f(top[0], top[1], top[2], 1.0)
    glVertex2f(0, h)
    glEnd()


def set_screen_projection(w: int, h: int) -> None:
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(0, w, 0, h, -1, 1)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()


def set_world_projection(model: DataModel, viewport_w: int, viewport_h: int) -> None:
    # Add a little padding around the data range.
    x_min, x_max = float(model.X_MIN), float(model.X_MAX)
    y_min, y_max = float(model.Y_MIN), float(model.Y_MAX)

    x_range = (x_max - x_min) if x_max > x_min else 1.0
    y_range = (y_max - y_min) if y_max > y_min else 1.0

    # Base padding (smaller so the data fills more of the plot area).
    # We intentionally *do not* expand to match viewport aspect; that expansion can
    # add a lot of empty space in wide viewports and make the data look too small.
    x_pad = max(1e-6, x_range * 0.04)
    y_pad = max(1e-6, y_range * 0.06)

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(x_min - x_pad, x_max + x_pad, y_min - y_pad, y_max + y_pad, -1, 1)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()


def draw_grid(model: DataModel) -> None:
    """Draw only the grid in world coordinates (no axes), so axes can be drawn fixed in screen-space."""
    grid_rgb = hex_to_rgb01(cfg.GRID_COLOR)

    x_min, x_max = float(model.X_MIN), float(model.X_MAX)
    y_min, y_max = float(model.Y_MIN), float(model.Y_MAX)

    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    glColor4f(grid_rgb[0], grid_rgb[1], grid_rgb[2], 0.28)
    glLineWidth(1.0)

    glBegin(GL_LINES)
    for i in range(1, 10):
        t = i / 10.0
        x = x_min + (x_max - x_min) * t
        glVertex2f(x, y_min)
        glVertex2f(x, y_max)

        y = y_min + (y_max - y_min) * t
        glVertex2f(x_min, y)
        glVertex2f(x_max, y)
    glEnd()


def draw_fixed_grid(*, x0: int, y0: int, w: int, h: int, divisions: int = 10) -> None:
    """Draw a screen-space grid inside a given rectangle (in plot-viewport pixels)."""
    grid_rgb = hex_to_rgb01(cfg.GRID_COLOR)

    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glColor4f(grid_rgb[0], grid_rgb[1], grid_rgb[2], 0.28)
    glLineWidth(1.0)

    x0f = float(x0)
    y0f = float(y0)
    x1f = float(x0 + w)
    y1f = float(y0 + h)

    glBegin(GL_LINES)
    for i in range(0, divisions + 1):
        t = i / max(divisions, 1)
        x = x0f + (x1f - x0f) * t
        y = y0f + (y1f - y0f) * t
        # vertical
        glVertex2f(x, y0f)
        glVertex2f(x, y1f)
        # horizontal
        glVertex2f(x0f, y)
        glVertex2f(x1f, y)
    glEnd()


def draw_fixed_axes(*, x0: int, y0: int, w: int, h: int, divisions: int = 10, thickness: float = 7.0) -> None:
    """Draw fixed x/y axes in screen-space for a given plot-rectangle.

    The axes start at (x0, y0) and extend to (x0+w, y0+h). This keeps the axes,
    grid, and data rectangle visually aligned.
    """
    axis_rgb = hex_to_rgb01(cfg.AXIS_COLOR)

    x0f = float(x0)
    y0f = float(y0)
    x1f = float(x0 + max(w, 1))
    y1f = float(y0 + max(h, 1))

    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    # Main axes as filled quads to avoid any join gaps.
    glColor4f(axis_rgb[0], axis_rgb[1], axis_rgb[2], 0.98)
    half = thickness * 0.5

    glBegin(GL_TRIANGLES)
    # x-axis quad: (x0, y0) -> (x0+w, y0)
    ax0, ay0 = x0f, y0f
    ax1, ay1 = x1f, y0f
    # quad corners
    q1 = (ax0, ay0 - half)
    q2 = (ax1, ay1 - half)
    q3 = (ax1, ay1 + half)
    q4 = (ax0, ay0 + half)
    glVertex2f(*q1)
    glVertex2f(*q2)
    glVertex2f(*q3)
    glVertex2f(*q1)
    glVertex2f(*q3)
    glVertex2f(*q4)

    # y-axis quad: (x0, y0) -> (x0, y0+h)
    bx0, by0 = x0f, y0f
    bx1, by1 = x0f, y1f
    r1 = (bx0 - half, by0)
    r2 = (bx0 + half, by0)
    r3 = (bx1 + half, by1)
    r4 = (bx1 - half, by1)
    glVertex2f(*r1)
    glVertex2f(*r2)
    glVertex2f(*r3)
    glVertex2f(*r1)
    glVertex2f(*r3)
    glVertex2f(*r4)

    # Joint square to guarantee no seam at the origin.
    j1 = (x0f - half, y0f - half)
    j2 = (x0f + half, y0f - half)
    j3 = (x0f + half, y0f + half)
    j4 = (x0f - half, y0f + half)
    glVertex2f(*j1)
    glVertex2f(*j2)
    glVertex2f(*j3)
    glVertex2f(*j1)
    glVertex2f(*j3)
    glVertex2f(*j4)
    glEnd()

    # Ticks
    glLineWidth(2.0)
    tick_len = 7.0
    tick_count = max(1, int(divisions))
    glBegin(GL_LINES)
    for i in range(1, tick_count):
        t = i / tick_count
        tx = x0f + (x1f - x0f) * t
        glVertex2f(tx, y0f)
        glVertex2f(tx, y0f + tick_len)
    for i in range(1, tick_count):
        t = i / tick_count
        ty = y0f + (y1f - y0f) * t
        glVertex2f(x0f, ty)
        glVertex2f(x0f + tick_len, ty)
    glEnd()


def draw_axes_and_grid(model: DataModel) -> None:
    grid_rgb = hex_to_rgb01(cfg.GRID_COLOR)
    axis_rgb = hex_to_rgb01(cfg.AXIS_COLOR)

    x_min, x_max = float(model.X_MIN), float(model.X_MAX)
    y_min, y_max = float(model.Y_MIN), float(model.Y_MAX)

    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    # Grid (fixed count)
    glColor4f(grid_rgb[0], grid_rgb[1], grid_rgb[2], 0.28)
    glLineWidth(1.0)

    glBegin(GL_LINES)
    for i in range(1, 10):
        t = i / 10.0
        x = x_min + (x_max - x_min) * t
        glVertex2f(x, y_min)
        glVertex2f(x, y_max)

        y = y_min + (y_max - y_min) * t
        glVertex2f(x_min, y)
        glVertex2f(x_max, y)
    glEnd()

    # Axes through (0,0) if visible, otherwise at mins.
    x_axis_y = 0.0 if y_min <= 0.0 <= y_max else y_min
    y_axis_x = 0.0 if x_min <= 0.0 <= x_max else x_min

    glColor4f(axis_rgb[0], axis_rgb[1], axis_rgb[2], 0.95)
    glLineWidth(2.5)
    glBegin(GL_LINES)
    glVertex2f(x_min, x_axis_y)
    glVertex2f(x_max, x_axis_y)
    glVertex2f(y_axis_x, y_min)
    glVertex2f(y_axis_x, y_max)
    glEnd()


def draw_points(model: DataModel) -> None:
    fill = hex_to_rgb01(cfg.POINT_FILL)
    outline = hex_to_rgb01(cfg.POINT_OUTLINE)

    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glEnable(GL_POINT_SMOOTH)

    # Outline
    glPointSize(8.0)
    glColor4f(outline[0], outline[1], outline[2], 1.0)
    glBegin(GL_POINTS)
    for x, y in zip(model.X, model.Y):
        glVertex2f(float(x), float(y))
    glEnd()

    # Fill
    glPointSize(5.0)
    glColor4f(fill[0], fill[1], fill[2], 1.0)
    glBegin(GL_POINTS)
    for x, y in zip(model.X, model.Y):
        glVertex2f(float(x), float(y))
    glEnd()


def draw_regression_line(model: DataModel, slope: float, intercept: float) -> None:
    line_rgb = hex_to_rgb01(cfg.LINE_COLOR)
    glow_rgb = hex_to_rgb01(cfg.LINE_GLOW)

    x1 = float(model.X_MIN)
    x2 = float(model.X_MAX)
    y1 = slope * x1 + intercept
    y2 = slope * x2 + intercept

    glEnable(GL_LINE_SMOOTH)

    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    # Glow pass
    glLineWidth(8.0)
    glColor4f(glow_rgb[0], glow_rgb[1], glow_rgb[2], 0.18)
    glBegin(GL_LINES)
    glVertex2f(x1, y1)
    glVertex2f(x2, y2)
    glEnd()

    # Main line
    glLineWidth(3.5)
    glColor4f(line_rgb[0], line_rgb[1], line_rgb[2], 1.0)
    glBegin(GL_LINES)
    glVertex2f(x1, y1)
    glVertex2f(x2, y2)
    glEnd()


def draw_residuals(model: DataModel, slope: float, intercept: float) -> None:
    res_rgb = hex_to_rgb01(cfg.RESIDUAL_COLOR)
    glLineWidth(1.0)
    glColor4f(res_rgb[0], res_rgb[1], res_rgb[2], 0.7)

    glBegin(GL_LINES)
    for x, y in zip(model.X, model.Y):
        y_pred = slope * float(x) + intercept
        glVertex2f(float(x), float(y))
        glVertex2f(float(x), float(y_pred))
    glEnd()


def draw_true_line(model: DataModel) -> None:
    # Use the target line from config as the "True Line".
    rgb = hex_to_rgb01(cfg.AXIS_COLOR)
    x1 = float(model.X_MIN)
    x2 = float(model.X_MAX)
    y1 = cfg.TRUE_S * x1 + cfg.TRUE_I
    y2 = cfg.TRUE_S * x2 + cfg.TRUE_I

    glEnable(GL_LINE_SMOOTH)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glLineWidth(2.5)
    glColor4f(rgb[0], rgb[1], rgb[2], 0.9)
    glBegin(GL_LINES)
    glVertex2f(x1, y1)
    glVertex2f(x2, y2)
    glEnd()


def draw_mse_panel(history: list[float]) -> None:
    # Draw a simple chart background + polyline in the current viewport.
    bg = (15 / 255.0, 23 / 255.0, 42 / 255.0)
    grid = (36 / 255.0, 52 / 255.0, 77 / 255.0)
    line = (249 / 255.0, 115 / 255.0, 22 / 255.0)

    # Background
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glColor4f(bg[0], bg[1], bg[2], 0.75)
    glBegin(GL_TRIANGLES)
    glVertex2f(0.0, 0.0)
    glVertex2f(1.0, 0.0)
    glVertex2f(1.0, 1.0)
    glVertex2f(0.0, 0.0)
    glVertex2f(1.0, 1.0)
    glVertex2f(0.0, 1.0)
    glEnd()

    # Grid lines
    glColor4f(grid[0], grid[1], grid[2], 0.45)
    glLineWidth(1.0)
    glBegin(GL_LINES)
    for i in range(1, 10):
        t = i / 10.0
        glVertex2f(t, 0.0)
        glVertex2f(t, 1.0)
        glVertex2f(0.0, t)
        glVertex2f(1.0, t)
    glEnd()

    if len(history) == 1:
        # Draw a single marker at the first point.
        x0 = 0.0
        y0 = 0.5
        glEnable(GL_POINT_SMOOTH)
        glPointSize(6.0)
        glColor4f(line[0], line[1], line[2], 0.9)
        glBegin(GL_POINTS)
        glVertex2f(float(x0), float(y0))
        glEnd()
        return
    if len(history) < 2:
        return

    y_max = max(history)
    y_min = min(history)
    y_range = max(y_max - y_min, 1e-6)

    glEnable(GL_LINE_SMOOTH)
    glLineWidth(2.0)
    glColor4f(line[0], line[1], line[2], 0.9)
    glBegin(GL_LINES)
    for i in range(1, len(history)):
        x0 = (i - 1) / (len(history) - 1)
        x1 = i / (len(history) - 1)
        y0 = (history[i - 1] - y_min) / y_range
        y1 = (history[i] - y_min) / y_range
        glVertex2f(float(x0), float(y0))
        glVertex2f(float(x1), float(y1))
    glEnd()


def make_optimizer(name: str):
    if name == "SGD":
        return SGD(cfg.LEARNING_RATE)
    if name == "Momentum":
        return Momentum(cfg.LEARNING_RATE, momentum=0.9)
    if name == "Adam":
        return Adam(cfg.LEARNING_RATE)
    return SGD(cfg.LEARNING_RATE)


def main() -> None:
    pygame.init()

    # Match the reference screenshot layout: tall window with room for an MSE panel and bottom buttons.
    w, h = 1450, 920
    pygame.display.set_caption("Linear Regression")

    # Better default visuals: multisampling (when supported by the driver)
    pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS, 1)
    pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLESAMPLES, 4)
    window_flags = pygame.DOUBLEBUF | pygame.OPENGL | pygame.RESIZABLE
    pygame.display.set_mode((w, h), window_flags)

    layout = Layout(w, h)
    ui = UIManager(layout)

    # Clear color doesn't matter much since we draw a gradient, but keep it sane.
    glClearColor(0.05, 0.07, 0.12, 1.0)

    glEnable(GL_MULTISAMPLE)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    model = DataModel(seed=42)
    tracker = MetricsTracker()

    slope, intercept = float(cfg.INITIAL_S), float(cfg.INITIAL_I)
    iteration = 0

    # Populate metrics immediately so the UI shows values before the first step.
    y_pred0 = slope * model.X + intercept
    tracker.update(model.Y, y_pred0, iteration)

    optimizer_name = "SGD"
    optimizer = make_optimizer(optimizer_name)

    running = True
    training = False
    show_residuals = True
    show_formulas = True

    # Tk style delay in ms.
    delay_ms = float(cfg.DELAY)
    step_accumulator = 0.0

    clock = pygame.time.Clock()

    while running:
        dt = clock.tick(60) / 1000.0
        mouse_pos = pygame.mouse.get_pos()  # (x, y) top-left origin
        click_down = False
        click_up = False

        # --- events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.VIDEORESIZE:
                # Recreate the GL surface at the new size and rebuild layout/UI.
                w, h = int(event.w), int(event.h)
                pygame.display.set_mode((w, h), window_flags)
                layout = Layout(w, h)
                ui = UIManager(layout)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                click_down = True
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                click_up = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    training = not training
                elif event.key == pygame.K_t:
                    training = True
                elif event.key == pygame.K_p:
                    training = False
                elif event.key == pygame.K_s:
                    training = False
                    if iteration < cfg.MAX_ITER:
                        slope, intercept, _ = compute_step(model, slope, intercept, optimizer)
                        y_pred = slope * model.X + intercept
                        tracker.update(model.Y, y_pred, iteration)
                        iteration += 1
                elif event.key == pygame.K_r:
                    training = False
                    slope, intercept = float(cfg.INITIAL_S), float(cfg.INITIAL_I)
                    iteration = 0
                    tracker.clear()
                    y_pred0 = slope * model.X + intercept
                    tracker.update(model.Y, y_pred0, iteration)
                    if hasattr(optimizer, "reset"):
                        optimizer.reset()
                    step_accumulator = 0.0
                elif event.key == pygame.K_n:
                    training = False
                    model.generate()
                    slope, intercept = float(cfg.INITIAL_S), float(cfg.INITIAL_I)
                    iteration = 0
                    tracker.clear()
                    y_pred0 = slope * model.X + intercept
                    tracker.update(model.Y, y_pred0, iteration)
                    if hasattr(optimizer, "reset"):
                        optimizer.reset()
                    step_accumulator = 0.0
                elif event.key == pygame.K_1:
                    optimizer_name = "SGD"
                    optimizer = make_optimizer(optimizer_name)
                elif event.key == pygame.K_2:
                    optimizer_name = "Momentum"
                    optimizer = make_optimizer(optimizer_name)
                elif event.key == pygame.K_3:
                    optimizer_name = "Adam"
                    optimizer = make_optimizer(optimizer_name)
                elif event.key == pygame.K_v:
                    show_residuals = not show_residuals
                elif event.key == pygame.K_f:
                    show_formulas = not show_formulas
                elif event.key in (pygame.K_LEFTBRACKET, pygame.K_MINUS, pygame.K_KP_MINUS):
                    # Slower: increase delay
                    delay_ms = min(200.0, float(delay_ms) + 5.0)
                    step_accumulator = 0.0
                elif event.key in (pygame.K_RIGHTBRACKET, pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                    # Faster: decrease delay
                    delay_ms = max(1.0, float(delay_ms) - 5.0)
                    step_accumulator = 0.0

        # --- UI overlays
        current_mse, _mae, _r2 = tracker.get_current()
        status_text = "Training" if training else "Ready to Start Training"
        actions = ui.render(
            mouse_pos=mouse_pos,
            mouse_down=click_down,
            mouse_up=click_up,
            training=training,
            delay_ms=delay_ms,
            show_residuals=show_residuals,
            show_formulas=show_formulas,
            optimizer_name=optimizer_name,
            mse_value=current_mse,
            slope=slope,
            intercept=intercept,
            status_text=status_text,
        )

        if actions.get("start"):
            training = True
        if actions.get("pause"):
            training = False
        if actions.get("step"):
            training = False
            if iteration < cfg.MAX_ITER:
                slope, intercept, _ = compute_step(model, slope, intercept, optimizer)
                y_pred = slope * model.X + intercept
                tracker.update(model.Y, y_pred, iteration)
                iteration += 1
        if actions.get("reset"):
            training = False
            slope, intercept = float(cfg.INITIAL_S), float(cfg.INITIAL_I)
            iteration = 0
            tracker.clear()
            y_pred0 = slope * model.X + intercept
            tracker.update(model.Y, y_pred0, iteration)
            if hasattr(optimizer, "reset"):
                optimizer.reset()
            step_accumulator = 0.0
        if actions.get("regen"):
            training = False
            model.generate()
            slope, intercept = float(cfg.INITIAL_S), float(cfg.INITIAL_I)
            iteration = 0
            tracker.clear()
            y_pred0 = slope * model.X + intercept
            tracker.update(model.Y, y_pred0, iteration)
            if hasattr(optimizer, "reset"):
                optimizer.reset()
            step_accumulator = 0.0
        if actions.get("toggle_formulas"):
            show_formulas = not show_formulas
        if "set_optimizer" in actions:
            optimizer_name = str(actions["set_optimizer"])
            optimizer = make_optimizer(optimizer_name)
            step_accumulator = 0.0
        if "set_delay_ms" in actions:
            delay_ms = float(actions["set_delay_ms"])
            step_accumulator = 0.0
        if "show_residuals" in actions:
            show_residuals = bool(actions["show_residuals"])

        # --- training step (time-based, driven by delay in ms)
        if training and iteration < cfg.MAX_ITER:
            steps_per_second = 1000.0 / max(delay_ms, 1.0)
            step_accumulator += dt * max(steps_per_second, 1.0)
            steps_this_frame = min(int(step_accumulator), 25)
            if steps_this_frame > 0:
                step_accumulator -= steps_this_frame
                for _ in range(steps_this_frame):
                    if iteration >= cfg.MAX_ITER:
                        training = False
                        break
                    slope, intercept, _ = compute_step(model, slope, intercept, optimizer)
                    y_pred = slope * model.X + intercept
                    tracker.update(model.Y, y_pred, iteration)
                    iteration += 1

        # --- render
        glClear(GL_COLOR_BUFFER_BIT)

        # Background for full window
        glViewport(0, 0, w, h)
        draw_gradient_background(w, h, cfg.BG_TOP, cfg.BG_BOTTOM)

        # Plot area (center)
        px, py, pw, ph = layout.to_gl_viewport(layout.plot_x, layout.plot_y_top, layout.plot_w, layout.plot_h)
        axes_origin = 52
        axes_thickness = 7.0
        divisions = 10
        inner_offset = int(axes_origin)

        # Inner data area (in window pixels).
        inner_w = max(1, pw - inner_offset - 10)
        inner_h = max(1, ph - inner_offset - 10)
        inner_x = px + inner_offset
        inner_y = py + inner_offset

        # Grid aligned to fixed axes (drawn in plot screen-space, clipped to the inner data rect).
        glViewport(px, py, pw, ph)
        set_screen_projection(pw, ph)
        draw_fixed_grid(x0=inner_offset, y0=inner_offset, w=inner_w, h=inner_h, divisions=divisions)

        # Draw world content inside the inset viewport so the fixed axes never overlap it.
        glViewport(inner_x, inner_y, inner_w, inner_h)
        set_world_projection(model, inner_w, inner_h)
        draw_points(model)
        if show_residuals:
            draw_residuals(model, slope, intercept)
        draw_true_line(model)
        draw_regression_line(model, slope, intercept)

        # Draw fixed axes last over the full plot viewport (no arrows, no corner gap).
        glViewport(px, py, pw, ph)
        set_screen_projection(pw, ph)
        draw_fixed_axes(x0=inner_offset, y0=inner_offset, w=inner_w, h=inner_h, divisions=divisions, thickness=axes_thickness)

        # MSE panel area
        mx, my, mw, mh = layout.to_gl_viewport(layout.mse_x, layout.mse_y_top, layout.mse_w, layout.mse_h)
        glViewport(mx, my, mw, mh)
        set_screen_projection(1, 1)
        draw_mse_panel(tracker.mse_history)

        # UI overlays (left/right/bottom)
        ui.draw_panels()

        # Title bar
        pygame.display.set_caption("Linear Regression")

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()