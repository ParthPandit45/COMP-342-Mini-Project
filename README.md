# Linear Regression (OpenGL)

This project runs **OpenGL-only**.

## Requirements
- Python 3.10+ (tested on 3.12)
- A machine with working OpenGL drivers

## Setup (Windows)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run
```powershell
python main.py
```

## Keyboard Shortcuts
- `Esc`: Quit
- `Space`: Start/Pause training
- `T`: Start training
- `P`: Pause training
- `S`: Step once
- `R`: Reset model
- `N`: Regenerate data
- `1`/`2`/`3`: Switch optimizer (SGD/Momentum/Adam)
- `V`: Toggle residuals
- `F`: Toggle formulas/legend
- `[` / `]`: Slower / Faster (adjust delay)

If you get an OpenGL/GL context error, update your GPU drivers.
