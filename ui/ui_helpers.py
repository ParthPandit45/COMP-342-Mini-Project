"""Legacy GUI helpers (deprecated).

The project now uses an OpenGL UI. This module remains only for backwards
compatibility and is not used by the OpenGL runtime.
"""


def create_rounded_rectangle(canvas, x1, y1, x2, y2, radius=20, **kwargs):
    """
    Create a rounded rectangle on a canvas.

    Args:
        canvas: A canvas-like object
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
    raise RuntimeError("Legacy GUI has been removed. Use the OpenGL UI (run main.py).")


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
    raise RuntimeError("Legacy GUI has been removed. Use the OpenGL UI (run main.py).")
