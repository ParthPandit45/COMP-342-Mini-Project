"""Legacy educational UI hooks (deprecated).

The educational side panel is now rendered directly by the OpenGL UI
implementation in `ui/opengl_ui.py`.
"""


class EducationalPanel:
    def __init__(self, parent_frame=None):
        self.parent = parent_frame

    def create_formula_panel(self) -> None:
        return

    def create_legend_panel(self) -> None:
        return

    def hide_all(self) -> None:
        return

    def show_all(self) -> None:
        return


class CanvasAnnotations:
    def __init__(self, canvas=None):
        self.canvas = canvas

    def clear_all(self) -> None:
        return

    def draw_convergence_indicator(self, is_converging: bool, convergence_rate: float) -> None:
        return
