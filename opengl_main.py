"""Compatibility wrapper (legacy).

The OpenGL application entrypoint is in `main.py`.

This module remains only so older commands like `python opengl_main.py` continue
to run, but new code should import from `main`.
"""

from __future__ import annotations

from main import main


if __name__ == "__main__":
    main()
