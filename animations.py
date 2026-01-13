"""Interactive animations and visual effects."""

import tkinter as tk
import config as cfg
import math


class AnimationEngine:
    """Handles animated visual effects."""
    
    def __init__(self, canvas):
        self.canvas = canvas
        self.particles = []
        self.highlight_ids = []
        
    def create_gradient_step_animation(self, x1, y1, x2, y2):
        """Animate the gradient step with a visual arrow."""
        arrow_id = self.canvas.create_line(
            x1, y1, x2, y2,
            fill="#fbbf24",
            width=3,
            arrow=tk.LAST,
            arrowshape=(16, 20, 6),
            tags="animation",
            dash=(4, 4)
        )
        self.canvas.after(800, lambda: self.canvas.delete(arrow_id))
        
    def pulse_point(self, x, y, color="#fbbf24"):
        """Create a pulsing effect at a point."""
        circles = []
        for i in range(3):
            size = 8 + i * 4
            circle = self.canvas.create_oval(
                x - size, y - size,
                x + size, y + size,
                outline=color,
                width=2,
                tags="pulse"
            )
            circles.append(circle)
            self.canvas.after(i * 100 + 300, lambda c=circle: self.canvas.delete(c))
    
    def show_error_contribution(self, points, errors):
        """Visualize which points contribute most to error."""
        if not points or len(errors) == 0:
            return
        
        max_error = max(abs(e) for e in errors)
        if max_error == 0:
            return
            
        for (px, py), error in zip(points, errors):
            normalized_error = abs(error) / max_error
            if normalized_error > 0.5:  # Only show significant errors
                # Create a colored ring around high-error points
                size = 6 + normalized_error * 4
                color = "#ef4444" if error > 0 else "#3b82f6"
                ring = self.canvas.create_oval(
                    px - size, py - size,
                    px + size, py + size,
                    outline=color,
                    width=2,
                    tags="error_highlight"
                )
                self.highlight_ids.append(ring)
    
    def clear_highlights(self):
        """Remove all highlight effects."""
        for hid in self.highlight_ids:
            self.canvas.delete(hid)
        self.highlight_ids.clear()
        self.canvas.delete("error_highlight")
        self.canvas.delete("pulse")
        self.canvas.delete("animation")


class VisualEffects:
    """Additional visual effects and indicators."""
    
    @staticmethod
    def create_gradient_arrow(canvas, model, old_slope, old_intercept, new_slope, new_intercept):
        """Draw an arrow showing the direction of gradient descent."""
        # Calculate positions for old and new lines at midpoint
        x_mid = (model.X_MIN + model.X_MAX) / 2
        
        y_old = old_slope * x_mid + old_intercept
        y_new = new_slope * x_mid + new_intercept
        
        px_mid, py_old = model.data_to_canvas(x_mid, y_old)
        _, py_new = model.data_to_canvas(x_mid, y_new)
        
        if abs(py_new - py_old) > 2:  # Only draw if movement is visible
            arrow = canvas.create_line(
                px_mid, py_old, px_mid, py_new,
                fill="#fbbf24",
                width=2,
                arrow=tk.LAST,
                arrowshape=(10, 12, 4),
                tags="gradient_arrow"
            )
            canvas.after(400, lambda: canvas.delete(arrow))
    
    @staticmethod
    def draw_best_fit_comparison(canvas, model, current_slope, current_intercept, true_slope, true_intercept):
        """Show comparison with true line if available."""
        if abs(true_slope - 1.5) < 0.01 and abs(true_intercept - 2.0) < 0.01:  # Only if we have true values
            y1_true = true_slope * model.X_MIN + true_intercept
            y2_true = true_slope * model.X_MAX + true_intercept
            
            p1x, p1y = model.data_to_canvas(model.X_MIN, y1_true)
            p2x, p2y = model.data_to_canvas(model.X_MAX, y2_true)
            
            true_line = canvas.create_line(
                p1x, p1y, p2x, p2y,
                fill="#10b981",
                width=2,
                dash=(8, 4),
                tags="true_line"
            )
            
            # Add label
            label_x = p2x - 80
            label_y = p2y - 10
            canvas.create_text(
                label_x, label_y,
                text="True Line",
                font=("Helvetica", 9),
                fill="#10b981",
                tags="true_line"
            )
