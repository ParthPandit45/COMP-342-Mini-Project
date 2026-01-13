"""Helper functions for creating smooth, rounded UI elements."""

import tkinter as tk


def create_rounded_rectangle(canvas, x1, y1, x2, y2, radius=20, **kwargs):
    """
    Create a rounded rectangle on a canvas.
    
    Args:
        canvas: The tkinter canvas
        x1, y1: Top-left coordinates
        x2, y2: Bottom-right coordinates
        radius: Corner radius
        **kwargs: Additional canvas item options (fill, outline, width, etc.)
    
    Returns:
        List of item IDs that make up the rounded rectangle
    """
    points = [
        x1 + radius, y1,
        x1 + radius, y1,
        x2 - radius, y1,
        x2 - radius, y1,
        x2, y1,
        x2, y1 + radius,
        x2, y1 + radius,
        x2, y2 - radius,
        x2, y2 - radius,
        x2, y2,
        x2 - radius, y2,
        x2 - radius, y2,
        x1 + radius, y2,
        x1 + radius, y2,
        x1, y2,
        x1, y2 - radius,
        x1, y2 - radius,
        x1, y1 + radius,
        x1, y1 + radius,
        x1, y1
    ]
    
    return canvas.create_polygon(points, smooth=True, **kwargs)


def create_rounded_frame(parent, bg_color, border_color, border_width=2, corner_radius=15, **pack_opts):
    """
    Create a frame with rounded corners using a canvas.
    
    Args:
        parent: Parent widget
        bg_color: Background color
        border_color: Border color
        border_width: Border width
        corner_radius: Radius of corners
        **pack_opts: Options to pass to pack()
    
    Returns:
        Tuple of (container_frame, content_frame) where content should be placed in content_frame
    """
    # Create container
    container = tk.Frame(parent, bg=parent.cget('bg'))
    
    # Create canvas for rounded background
    canvas = tk.Canvas(container, highlightthickness=0, bg=parent.cget('bg'))
    canvas.pack(fill="both", expand=True)
    
    # Content frame
    content_frame = tk.Frame(container, bg=bg_color)
    
    # Draw rounded rectangle on canvas update
    def draw_rounded_bg(event=None):
        canvas.delete("rounded_bg")
        width = canvas.winfo_width()
        height = canvas.winfo_height()
        
        # Draw border
        if border_width > 0:
            create_rounded_rectangle(
                canvas, 0, 0, width, height,
                radius=corner_radius,
                fill=bg_color,
                outline=border_color,
                width=border_width,
                tags="rounded_bg"
            )
        else:
            create_rounded_rectangle(
                canvas, 0, 0, width, height,
                radius=corner_radius,
                fill=bg_color,
                outline=bg_color,
                tags="rounded_bg"
            )
    
    canvas.bind("<Configure>", draw_rounded_bg)
    
    # Place content frame on top
    content_frame.place(relx=0.5, rely=0.5, anchor="center", relwidth=0.95, relheight=0.9)
    
    if pack_opts:
        container.pack(**pack_opts)
    
    return container, content_frame


def create_smooth_button(parent, text, command, bg_color, fg_color, hover_color, **pack_opts):
    """
    Create a modern button with rounded corners and hover effect.
    
    Args:
        parent: Parent widget
        text: Button text
        command: Command to execute on click
        bg_color: Background color
        fg_color: Text color
        hover_color: Color on hover
        **pack_opts: Options to pass to pack()
    
    Returns:
        Button widget
    """
    btn = tk.Button(
        parent,
        text=text,
        command=command,
        bg=bg_color,
        fg=fg_color,
        activebackground=hover_color,
        activeforeground=fg_color,
        font=("Helvetica", 10, "bold"),
        bd=0,
        padx=15,
        pady=8,
        cursor="hand2",
        relief=tk.FLAT
    )
    
    # Hover effects
    def on_enter(e):
        btn.config(bg=hover_color)
    
    def on_leave(e):
        btn.config(bg=bg_color)
    
    btn.bind("<Enter>", on_enter)
    btn.bind("<Leave>", on_leave)
    
    if pack_opts:
        btn.pack(**pack_opts)
    
    return btn
