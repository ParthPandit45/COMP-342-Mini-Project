"""Lightweight UI overlays for the OpenGL version.

This intentionally avoids external GUI libs (imgui) and instead draws UI via pygame
surfaces uploaded to OpenGL textures.
"""

from __future__ import annotations

from dataclasses import dataclass

import pygame
from OpenGL.GL import (
    GL_BLEND,
    GL_LINEAR,
    GL_MODELVIEW,
    GL_ONE_MINUS_SRC_ALPHA,
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
    glColor4f,
    glDisable,
    glEnable,
    glEnd,
    glGenTextures,
    glLoadIdentity,
    glMatrixMode,
    glOrtho,
    glTexCoord2f,
    glTexImage2D,
    glTexParameteri,
    glTexSubImage2D,
    glVertex2f,
    glViewport,
)


@dataclass(frozen=True)
class Layout:
    window_w: int
    window_h: int

    # Outer padding around all regions (kept small for a tight fit).
    padding: int = 8
    # Horizontal space between left/plot/right columns.
    gutter: int = 8
    # Vertical space between plot and MSE panel.
    gap_plot_mse: int = 8

    left_w: int = 220
    right_w: int = 300

    mse_h: int = 180
    status_h: int = 28
    buttons_h: int = 60

    @property
    def bottom_h(self) -> int:
        return self.status_h + self.buttons_h

    @property
    def left_x(self) -> int:
        return self.padding

    @property
    def right_x(self) -> int:
        return max(self.padding, self.window_w - self.padding - self.right_w)

    @property
    def bottom_x(self) -> int:
        return self.padding

    @property
    def bottom_w(self) -> int:
        return max(1, self.window_w - 2 * self.padding)

    @property
    def plot_h(self) -> int:
        # Total vertical layout (top-to-bottom):
        # padding + plot + gap + mse + bottom_bar + padding
        needed = 2 * self.padding + self.gap_plot_mse + self.mse_h + self.bottom_h
        return max(1, self.window_h - needed)

    @property
    def plot_y_top(self) -> int:
        return self.padding

    @property
    def plot_x(self) -> int:
        return self.left_x + self.left_w + self.gutter

    @property
    def plot_w(self) -> int:
        # Fill the space between left and right panels with gutters.
        w = self.right_x - self.gutter - self.plot_x
        return max(1, w)

    @property
    def side_y_top(self) -> int:
        return self.plot_y_top

    @property
    def side_h(self) -> int:
        # Make the side panels span the plot + the gap + the MSE panel for a tighter, more cohesive layout.
        return self.plot_h + self.gap_plot_mse + self.mse_h

    @property
    def mse_y_top(self) -> int:
        return self.plot_y_top + self.plot_h + self.gap_plot_mse

    @property
    def mse_x(self) -> int:
        return self.plot_x

    @property
    def mse_w(self) -> int:
        return self.plot_w

    @property
    def status_y_top(self) -> int:
        return self.mse_y_top + self.mse_h

    @property
    def buttons_y_top(self) -> int:
        return self.status_y_top + self.status_h

    def to_gl_viewport(self, x: int, y_top: int, w: int, h: int) -> tuple[int, int, int, int]:
        # OpenGL viewport expects bottom-left origin.
        return (x, self.window_h - (y_top + h), w, h)


class GLTextureSurface:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.surface = pygame.Surface((width, height), pygame.SRCALPHA)

        self.texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(
            GL_TEXTURE_2D,
            0,
            GL_RGBA,
            width,
            height,
            0,
            GL_RGBA,
            GL_UNSIGNED_BYTE,
            None,
        )

    def upload(self) -> None:
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        pixel_data = pygame.image.tostring(self.surface, "RGBA", True)
        glTexSubImage2D(
            GL_TEXTURE_2D,
            0,
            0,
            0,
            self.width,
            self.height,
            GL_RGBA,
            GL_UNSIGNED_BYTE,
            pixel_data,
        )

    def draw(self, x: int, y_top: int, layout: Layout) -> None:
        # Draw texture as a screen-space quad.
        glViewport(0, 0, layout.window_w, layout.window_h)

        # Ensure we are in pixel-space regardless of what the plot/MSE viewport set.
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, layout.window_w, 0, layout.window_h, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, self.texture_id)

        # Convert to screen-space with bottom-left origin.
        y_bottom = layout.window_h - (y_top + self.height)
        x1, y1 = float(x), float(y_bottom)
        x2, y2 = float(x + self.width), float(y_bottom + self.height)

        glColor4f(1.0, 1.0, 1.0, 1.0)
        glBegin(GL_TRIANGLES)
        # triangle 1
        glTexCoord2f(0.0, 0.0)
        glVertex2f(x1, y1)
        glTexCoord2f(1.0, 0.0)
        glVertex2f(x2, y1)
        glTexCoord2f(1.0, 1.0)
        glVertex2f(x2, y2)
        # triangle 2
        glTexCoord2f(0.0, 0.0)
        glVertex2f(x1, y1)
        glTexCoord2f(1.0, 1.0)
        glVertex2f(x2, y2)
        glTexCoord2f(0.0, 1.0)
        glVertex2f(x1, y2)
        glEnd()

        glDisable(GL_TEXTURE_2D)


class Widgets:
    def __init__(self, font: pygame.font.Font, font_small: pygame.font.Font):
        self.font = font
        self.font_small = font_small
        self.dragging_slider = False
        self.dragging_vslider = False

    def text(self, surf: pygame.Surface, text: str, x: int, y: int, *, small: bool = False, color=(226, 232, 240)) -> None:
        f = self.font_small if small else self.font
        s = f.render(text, True, color)
        surf.blit(s, (x, y))

    def button(self, surf: pygame.Surface, label: str, rect: pygame.Rect, mouse_pos: tuple[int, int], click: bool,
               *, bg=(14, 165, 233), hover_bg=(2, 132, 199), text_color=(255, 255, 255), radius=6) -> bool:
        hover = rect.collidepoint(mouse_pos)
        color = hover_bg if hover else bg
        pygame.draw.rect(surf, (*color, 255), rect, border_radius=radius)
        pygame.draw.rect(surf, (7, 89, 133, 255), rect, width=2, border_radius=radius)
        ts = self.font_small.render(label, True, text_color)
        surf.blit(ts, (rect.x + (rect.w - ts.get_width()) // 2, rect.y + (rect.h - ts.get_height()) // 2))
        return hover and click

    def checkbox(self, surf: pygame.Surface, label: str, rect: pygame.Rect, checked: bool, mouse_pos: tuple[int, int], click: bool) -> bool:
        hover = rect.collidepoint(mouse_pos)
        box = pygame.Rect(rect.x, rect.y, 18, 18)
        pygame.draw.rect(surf, (15, 23, 42, 255), rect, border_radius=6)
        pygame.draw.rect(surf, (94, 234, 212, 255) if hover else (148, 163, 184, 255), box, width=2, border_radius=3)
        if checked:
            pygame.draw.rect(surf, (94, 234, 212, 255), box.inflate(-6, -6), border_radius=2)
        self.text(surf, label, rect.x + 26, rect.y - 2, small=True)
        return checked if not (hover and click) else (not checked)

    def hslider(self, surf: pygame.Surface, label: str, rect: pygame.Rect, value: float, vmin: float, vmax: float,
                mouse_pos: tuple[int, int], mouse_down: bool, mouse_up: bool) -> tuple[bool, float]:
        pygame.draw.rect(surf, (15, 23, 42, 255), rect, border_radius=8)
        track = pygame.Rect(rect.x + 12, rect.y + rect.h // 2, rect.w - 24, 6)
        pygame.draw.rect(surf, (2, 6, 23, 255), track, border_radius=4)

        t = 0.0 if vmax == vmin else (value - vmin) / (vmax - vmin)
        t = max(0.0, min(1.0, float(t)))
        knob_x = int(track.x + t * track.w)
        knob = pygame.Rect(knob_x - 6, track.y - 6, 12, 18)
        pygame.draw.rect(surf, (94, 234, 212, 255), knob, border_radius=4)
        self.text(surf, f"{label}: {value:.0f}", rect.x + 12, rect.y + 8, small=True)

        changed = False
        if mouse_down and (knob.collidepoint(mouse_pos) or track.collidepoint(mouse_pos)):
            self.dragging_slider = True
        if mouse_up:
            self.dragging_slider = False
        if self.dragging_slider:
            mx = mouse_pos[0]
            new_t = (mx - track.x) / max(track.w, 1)
            new_t = max(0.0, min(1.0, float(new_t)))
            new_value = vmin + new_t * (vmax - vmin)
            if abs(new_value - value) > 1e-6:
                value = new_value
                changed = True
        return changed, float(value)

    def vslider(self, surf: pygame.Surface, label: str, rect: pygame.Rect, value: float, vmin: float, vmax: float,
                mouse_pos: tuple[int, int], mouse_down: bool, mouse_up: bool) -> tuple[bool, float]:
        # Vertical slider: value increases upwards.
        pygame.draw.rect(surf, (15, 23, 42, 255), rect, border_radius=8)
        track = pygame.Rect(rect.x + rect.w // 2 - 3, rect.y + 12, 6, rect.h - 24)
        pygame.draw.rect(surf, (2, 6, 23, 255), track, border_radius=4)

        t = 0.0 if vmax == vmin else (value - vmin) / (vmax - vmin)
        t = max(0.0, min(1.0, float(t)))
        knob_y = int(track.y + (1.0 - t) * track.h)
        knob = pygame.Rect(track.x - 6, knob_y - 6, 18, 12)
        pygame.draw.rect(surf, (94, 234, 212, 255), knob, border_radius=4)

        self.text(surf, label, rect.x, rect.y - 20, small=True)

        changed = False
        if mouse_down and (knob.collidepoint(mouse_pos) or track.collidepoint(mouse_pos)):
            self.dragging_vslider = True
        if mouse_up:
            self.dragging_vslider = False
        if self.dragging_vslider:
            my = mouse_pos[1]
            new_t = 1.0 - ((my - track.y) / max(track.h, 1))
            new_t = max(0.0, min(1.0, float(new_t)))
            new_value = vmin + new_t * (vmax - vmin)
            if abs(new_value - value) > 1e-6:
                value = new_value
                changed = True
        return changed, float(value)


class UIManager:
    def __init__(self, layout: Layout):
        self.layout = layout

        self.left = GLTextureSurface(layout.left_w, layout.side_h)
        self.right = GLTextureSurface(layout.right_w, layout.side_h)
        self.bottom = GLTextureSurface(layout.bottom_w, layout.bottom_h)

        self.font = pygame.font.SysFont("Segoe UI", 18)
        self.font_small = pygame.font.SysFont("Segoe UI", 14)
        self.widgets = Widgets(self.font, self.font_small)

        self._optimizer_open = False

    def draw_panels(self) -> None:
        self.left.draw(self.layout.left_x, self.layout.side_y_top, self.layout)
        self.right.draw(self.layout.right_x, self.layout.side_y_top, self.layout)
        self.bottom.draw(self.layout.bottom_x, self.layout.status_y_top, self.layout)

    def render(
        self,
        *,
        mouse_pos: tuple[int, int],
        mouse_down: bool,
        mouse_up: bool,
        training: bool,
        delay_ms: float,
        show_residuals: bool,
        show_formulas: bool,
        optimizer_name: str,
        mse_value: float,
        slope: float,
        intercept: float,
        status_text: str,
    ) -> dict:
        actions: dict[str, object] = {}

        # --- LEFT PANEL
        ls = self.left.surface
        ls.fill((0, 0, 0, 0))
        pygame.draw.rect(ls, (15, 23, 42, 245), pygame.Rect(0, 0, self.layout.left_w, self.layout.side_h))

        self.widgets.text(ls, "Training Controls", 14, 12, small=True, color=(203, 213, 225))

        # Delay slider
        slider_rect = pygame.Rect(20, 60, 40, 260)
        changed, delay_ms = self.widgets.vslider(
            ls,
            "Delay (ms) [ / ]",
            slider_rect,
            float(delay_ms),
            1.0,
            200.0,
            self._left_local(mouse_pos),
            mouse_down and self._in_left(mouse_pos),
            mouse_up,
        )
        if changed:
            actions["set_delay_ms"] = delay_ms

        show_residuals = self.widgets.checkbox(
            ls,
            "Show Residuals [V]",
            pygame.Rect(14, 340, self.layout.left_w - 28, 22),
            show_residuals,
            self._left_local(mouse_pos),
            mouse_down and self._in_left(mouse_pos),
        )
        actions["show_residuals"] = show_residuals

        self.widgets.text(ls, "Visual Aids", 14, 390, small=True, color=(203, 213, 225))
        if self.widgets.button(
            ls,
            "Toggle Formulas [F]",
            pygame.Rect(14, 420, self.layout.left_w - 28, 34),
            self._left_local(mouse_pos),
            mouse_down and self._in_left(mouse_pos),
            bg=(139, 92, 246),
            hover_bg=(124, 58, 237),
        ):
            actions["toggle_formulas"] = True

        self.left.upload()

        # --- RIGHT PANEL
        rs = self.right.surface
        rs.fill((0, 0, 0, 0))
        pygame.draw.rect(rs, (15, 23, 42, 245), pygame.Rect(0, 0, self.layout.right_w, self.layout.side_h))

        self.widgets.text(rs, "Algorithm Reference", 14, 12, small=True, color=(203, 213, 225))

        y = 44
        if show_formulas:
            self.widgets.text(rs, "Gradient Descent Equations", 14, y, small=True, color=(251, 191, 36))
            y += 22
            self.widgets.text(rs, "Slope Update:", 14, y, small=True)
            y += 18
            self.widgets.text(rs, "s = s - (a/n) * \u03a3(error * x)", 14, y, small=True, color=(148, 163, 184))
            y += 22
            self.widgets.text(rs, "Intercept Update:", 14, y, small=True)
            y += 18
            self.widgets.text(rs, "i = i - (a/n) * \u03a3(error)", 14, y, small=True, color=(148, 163, 184))
            y += 22
            self.widgets.text(rs, "Cost Function:", 14, y, small=True)
            y += 18
            self.widgets.text(rs, "MSE = (1/n) * \u03a3(y - \u0177)\u00b2", 14, y, small=True, color=(148, 163, 184))
            y += 26

            # Legend box
            legend_rect = pygame.Rect(14, y, self.layout.right_w - 28, 130)
            pygame.draw.rect(rs, (30, 41, 59, 220), legend_rect, border_radius=8)
            self.widgets.text(rs, "Legend", legend_rect.x + 10, legend_rect.y + 8, small=True)

            ly = legend_rect.y + 34
            items = [
                ((96, 165, 250), "Training Data Points"),
                ((249, 115, 22), "Predicted Regression Line"),
                ((203, 213, 225), "Error (Residuals)"),
                ((94, 234, 212), "True Target Line"),
            ]
            for color, label in items:
                pygame.draw.circle(rs, color, (legend_rect.x + 16, ly + 6), 4)
                self.widgets.text(rs, label, legend_rect.x + 28, ly, small=True, color=(226, 232, 240))
                ly += 22

            y = legend_rect.y + legend_rect.h + 18

        # Model settings
        self.widgets.text(rs, "Model Settings", 14, y, small=True, color=(203, 213, 225))
        y += 28
        self.widgets.text(rs, "Optimizer [1/2/3]", 14, y, small=True)
        y += 18

        opt_rect = pygame.Rect(14, y, self.layout.right_w - 28, 34)
        local_mouse = self._right_local(mouse_pos)
        clicked = mouse_down and self._in_right(mouse_pos) and opt_rect.collidepoint(local_mouse)
        if clicked:
            self._optimizer_open = not self._optimizer_open

        pygame.draw.rect(rs, (14, 165, 233, 255), opt_rect, border_radius=6)
        self.widgets.text(rs, optimizer_name, opt_rect.x + 14, opt_rect.y + 8, small=True, color=(255, 255, 255))

        if self._optimizer_open:
            menu = pygame.Rect(opt_rect.x, opt_rect.y + opt_rect.h + 6, opt_rect.w, 110)
            pygame.draw.rect(rs, (30, 41, 59, 240), menu, border_radius=8)
            options = ["SGD", "Momentum", "Adam"]
            oy = menu.y + 8
            for opt in options:
                row = pygame.Rect(menu.x + 8, oy, menu.w - 16, 28)
                hover = row.collidepoint(local_mouse)
                pygame.draw.rect(rs, (51, 65, 85, 220) if hover else (30, 41, 59, 0), row, border_radius=6)
                self.widgets.text(rs, opt, row.x + 10, row.y + 6, small=True)
                if mouse_down and self._in_right(mouse_pos) and row.collidepoint(local_mouse):
                    actions["set_optimizer"] = opt
                    self._optimizer_open = False
                oy += 32

        self.right.upload()

        # --- BOTTOM BAR (status + buttons)
        bs = self.bottom.surface
        bs.fill((0, 0, 0, 0))
        pygame.draw.rect(bs, (15, 23, 42, 255), pygame.Rect(0, 0, self.layout.bottom_w, self.layout.bottom_h))

        # Status strip
        pygame.draw.rect(bs, (30, 41, 59, 220), pygame.Rect(0, 0, self.layout.bottom_w, self.layout.status_h))
        self.widgets.text(bs, f"MSE: {mse_value:.2f}", 14, 6, small=True)
        self.widgets.text(bs, status_text, 120, 6, small=True, color=(203, 213, 225))
        self.widgets.text(bs, f"Slope: {slope:.3f}", 520, 6, small=True, color=(203, 213, 225))
        self.widgets.text(bs, f"Intercept: {intercept:.3f}", 660, 6, small=True, color=(203, 213, 225))
        self.widgets.text(bs, "Idle" if not training else "Running", 820, 6, small=True, color=(249, 115, 22) if not training else (16, 185, 129))

        # Buttons row
        by = self.layout.status_h + 10
        bx = 14
        bw = 190
        bh = 36
        gap = 12

        local = self._bottom_local(mouse_pos)
        in_bottom = self._in_bottom(mouse_pos)

        if self.widgets.button(bs, "Start Training [T/Space]", pygame.Rect(bx, by, bw, bh), local, mouse_down and in_bottom):
            actions["start"] = True
        bx += bw + gap
        if self.widgets.button(bs, "Pause Training [P/Space]", pygame.Rect(bx, by, bw, bh), local, mouse_down and in_bottom):
            actions["pause"] = True
        bx += bw + gap
        if self.widgets.button(bs, "Step Once [S]", pygame.Rect(bx, by, bw, bh), local, mouse_down and in_bottom):
            actions["step"] = True
        bx += bw + gap
        if self.widgets.button(bs, "Reset Model [R]", pygame.Rect(bx, by, bw, bh), local, mouse_down and in_bottom):
            actions["reset"] = True
        bx += bw + gap
        if self.widgets.button(bs, "Regenerate Data [N]", pygame.Rect(bx, by, bw + 40, bh), local, mouse_down and in_bottom):
            actions["regen"] = True

        self.bottom.upload()

        return actions

    def _in_left(self, mouse_pos: tuple[int, int]) -> bool:
        x, y = mouse_pos
        return (
            self.layout.left_x <= x < self.layout.left_x + self.layout.left_w
            and self.layout.side_y_top <= y < self.layout.side_y_top + self.layout.side_h
        )

    def _left_local(self, mouse_pos: tuple[int, int]) -> tuple[int, int]:
        x, y = mouse_pos
        return (x - self.layout.left_x, y - self.layout.side_y_top)

    def _in_right(self, mouse_pos: tuple[int, int]) -> bool:
        x, y = mouse_pos
        return (
            self.layout.right_x <= x < self.layout.right_x + self.layout.right_w
            and self.layout.side_y_top <= y < self.layout.side_y_top + self.layout.side_h
        )

    def _right_local(self, mouse_pos: tuple[int, int]) -> tuple[int, int]:
        x, y = mouse_pos
        return (x - self.layout.right_x, y - self.layout.side_y_top)

    def _in_bottom(self, mouse_pos: tuple[int, int]) -> bool:
        x, y = mouse_pos
        return (
            self.layout.bottom_x <= x < self.layout.bottom_x + self.layout.bottom_w
            and self.layout.status_y_top <= y < self.layout.status_y_top + self.layout.bottom_h
        )

    def _bottom_local(self, mouse_pos: tuple[int, int]) -> tuple[int, int]:
        x, y = mouse_pos
        return (x - self.layout.bottom_x, y - self.layout.status_y_top)
