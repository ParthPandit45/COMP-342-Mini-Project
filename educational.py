"""Educational overlays and annotations for the visualization."""

import tkinter as tk
import config as cfg


class EducationalPanel:
    """Manages educational content in separate UI panels."""
    
    def __init__(self, parent_frame):
        self.parent = parent_frame
        self.formula_frame = None
        self.legend_frame = None
        
    def create_formula_panel(self):
        """Create a panel displaying gradient descent formulas."""
        if self.formula_frame:
            self.formula_frame.destroy()
            
        self.formula_frame = tk.Frame(self.parent, bg=cfg.BG_BOTTOM)
        self.formula_frame.pack(fill="x", pady=(0, 15))
        
        # Title
        tk.Label(self.formula_frame, text="Gradient Descent Equations",
                font=("Helvetica", 10, "bold"), fg="#fbbf24", bg=cfg.BG_BOTTOM).pack(anchor="w")
        
        # Formulas with explanations
        formulas = [
            ("Slope Update:", "θ₁ := θ₁ - α·(2/n)·Σ(error·x)"),
            ("Intercept Update:", "θ₀ := θ₀ - α·(2/n)·Σ(error)"),
            ("Cost Function:", "MSE = (1/n)·Σ(ŷ - y)²")
        ]
        
        for label, formula in formulas:
            tk.Label(self.formula_frame, text=label,
                    font=("Helvetica", 8), fg="#94a3b8", bg=cfg.BG_BOTTOM).pack(anchor="w", pady=(5, 0))
            tk.Label(self.formula_frame, text=formula,
                    font=("Courier", 9), fg="#a5b4fc", bg=cfg.BG_BOTTOM).pack(anchor="w", padx=(10, 0))
    
    def create_legend_panel(self):
        """Create a legend panel explaining visual elements."""
        if self.legend_frame:
            self.legend_frame.destroy()
            
        self.legend_frame = tk.Frame(self.parent, bg="#1a2332", relief=tk.FLAT, borderwidth=0)
        self.legend_frame.pack(fill="x", pady=(0, 15), padx=5)
        
        # Create inner frame with padding for rounded look
        inner_frame = tk.Frame(self.legend_frame, bg="#1e293b")
        inner_frame.pack(fill="both", expand=True, padx=3, pady=3)
        
        # Title with better styling
        title_frame = tk.Frame(inner_frame, bg="#2d3748")
        title_frame.pack(fill="x", padx=2, pady=2)
        tk.Label(title_frame, text="Legend",
                font=("Helvetica", 10, "bold"), fg=cfg.TEXT_COLOR, bg="#2d3748").pack(pady=5)
        
        # Legend items with better spacing
        items = [
            ("● ", cfg.POINT_FILL, "Training Data Points"),
            ("━ ", cfg.LINE_COLOR, "Predicted Regression Line"),
            ("┄ ", cfg.RESIDUAL_COLOR, "Error (Residuals)"),
            ("┄ ", "#10b981", "True Target Line")
        ]
        
        content_frame = tk.Frame(inner_frame, bg="#1e293b")
        content_frame.pack(fill="both", expand=True, padx=8, pady=5)
        
        for symbol, color, description in items:
            item_frame = tk.Frame(content_frame, bg="#1e293b")
            item_frame.pack(fill="x", pady=3)
            
            tk.Label(item_frame, text=symbol, fg=color, bg="#1e293b",
                    font=("Helvetica", 14, "bold")).pack(side=tk.LEFT)
            tk.Label(item_frame, text=description, fg=cfg.TEXT_COLOR, bg="#1e293b",
                    font=("Helvetica", 8)).pack(side=tk.LEFT, padx=(8, 0))
        
        tk.Label(content_frame, text="", bg="#1e293b").pack(pady=2)
    
    def hide_all(self):
        """Hide all educational panels."""
        if self.formula_frame:
            self.formula_frame.pack_forget()
        if self.legend_frame:
            self.legend_frame.pack_forget()
    
    def show_all(self):
        """Show all educational panels."""
        if self.formula_frame:
            self.formula_frame.pack(fill="x", pady=(0, 15))
        if self.legend_frame:
            self.legend_frame.pack(fill="x", pady=(0, 15), padx=5)


class CanvasAnnotations:
    """Manages canvas-based annotations only."""
    
    def __init__(self, canvas):
        self.canvas = canvas
        self.annotation_ids = []
    
    def clear_all(self):
        """Clear all canvas annotations."""
        for aid in self.annotation_ids:
            self.canvas.delete(aid)
        self.annotation_ids.clear()
    
    def draw_convergence_indicator(self, is_converging, convergence_rate):
        """Show visual indicator of convergence status."""
        x_pos = cfg.CANVAS_W // 2
        y_pos = cfg.CANVAS_H - cfg.PAD + 25
        
        if is_converging:
            color = "#10b981"
            status = "✓ Converging"
        else:
            color = "#ef4444"
            status = "⚠ Adjusting"
        
        text_id = self.canvas.create_text(
            x_pos, y_pos,
            text=f"{status} (Rate: {convergence_rate:.4f})",
            font=("Helvetica", 10, "bold"),
            fill=color,
            tags="convergence"
        )
        self.annotation_ids.append(text_id)
