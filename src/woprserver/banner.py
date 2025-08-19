import os
import shutil
import sys
from multiprocessing import current_process

# ===== config =====
GAP = 6  # space between planet and title
ENV_ENABLE = "WORP_BANNER"
# ==================

# Your chosen "BETTER" planet ASCII (multi-line)
PLANET = r"""
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⠀⡀⢀⠀⢠⠀⠀⠀⠀⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⢠⢤⣀⠀⠀⠀⠈⣆⢧⠈⡆⢸⠀⠀⠀⢰⢡⠇⠀⠀⠀⠀⠀⡀⠀⠀⠀⠀⠀
⠀⠀⠀⢀⠀⠀⣯⢀⣨⠃⠀⠀⠀⠸⡜⣄⣣⢸⠀⠀⠀⡜⡌⠀⠀⠀⠀⢀⡜⡁⠀⠀⠀⠀⠀
⠀⠀⠙⢮⡳⢄⠈⠁⠀⢠⠴⠍⣛⣚⣣⢳⢽⡀⣏⣲⣀⢧⡥⠤⠶⢤⣠⢎⠜⠁⠀⠀⠀⠀⠀
⠀⠠⣀⠀⠙⢦⡑⢄⢀⣾⣧⡎⠁⠀⠙⡎⡇⡇⡇⠹⢪⣀⡓⣦⢀⣼⣵⠋⢀⠴⣊⠔⠁⠀⠀
⠀⠀⠈⠑⢦⣀⠙⣲⣝⢭⡚⠃⠀⠀⠀⠸⠸ __        _____  ____  ____                              
⠀⠀⠀⠀⠀⠈⣷⢯⣨⠷⣝⠦⠀⠀⠀⠀⠀ \ \      / / _ \|  _ \|  _ \ ___  ___ _ ____   _____ _ __ 
⠀⠀⠀⠀⠀⢀⡞⢠⠾⠓⢮⠁⠀⠀⠀⠀⠀  \ \ /\ / / | | | |_) | |_) / __|/ _ \ '__\ \ / / _ \ '__|
⠀⠀⢀⡤⣀⢈⡷⠻⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀  \ V  V /| |_| |  __/|  _ <\__ \  __/ |   \ V /  __/ |   
⠀⠀⢻⣀⡼⢘⣧⢀⡟⠉⠀⠀⠀⠀⠀⠀⠀⠀   \_/\_/  \___/|_|   |_| \_\___/\___|_|    \_/ \___|_|   
⠀⠀⠀⠉⠀⢿⡀⠈⠧⡤⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣸⠇⣹⣦⠇⠀⠀------------------------------------------>⠀⠀⠀
⠀⠀⠀⠀⠀⠸⢤⡴⢺⡧⣴⡶⢗⡣⠀⡀⠀⠀⠀⡄⠀⢀⣄⠢⣔⡞⣤⠦⡇⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⣀⡤⣖⣯⡗⣪⢽⡻⣅⠀⣜⡜⠀⠀⠀⠸⡜⡌⣮⡣⡙⢗⢏⡽⠁⠰⡏⠙⡆⠀⠀
⠀⠀⣒⡭⠖⣋⡥⣞⣿⡚⠉⠉⢉⢟⣞⣀⣀⣀⠐⢦⢵⠹⡍⢳⡝⢮⡷⢝⢦⡀⠉⠙⠁⠀⠀
⠐⠊⢡⠴⠚⠕⠋⠹⣍⡉⠹⢧⢫⢯⣀⣄⣀⠈⣹⢯⣀⣧⢹⡉⠙⢦⠙⣄⠑⢌⠲⣄⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠘⠧⡴⣳⣃⣸⠦⠴⠖⢾⣥⠞⠛⠘⣆⢳⡀⠈⠳⡈⠳⡄⠁⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⢀⡜⡱⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⡄⢣⠀⠀⠉⠀⠈⠂⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⢀⠞⡼⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠹⡀⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
"""

# Big "WOPRserver" text (ASCII figlet style)
TITLE = r"""

                                                           
"""

# --- color helpers ---
RESET = "\x1b[0m"
COLORS_WARM = ["\x1b[38;5;197m", "\x1b[38;5;203m", "\x1b[38;5;214m", "\x1b[38;5;220m"]
COLORS_COOL = ["\x1b[38;5;51m", "\x1b[38;5;45m", "\x1b[38;5;69m", "\x1b[38;5;111m"]

def _use_color() -> bool:
    if os.getenv("NO_COLOR"):
        return False
    try:
        return sys.stdout.isatty()
    except Exception:
        return False

def _colorize_lines(lines, palette):
    if not _use_color():
        return lines
    out = []
    for i, ln in enumerate(lines):
        out.append(palette[i % len(palette)] + ln + RESET)
    return out

def _pad(lines, width):
    return [ln + " " * max(0, width - len(ln)) for ln in lines]

def _side_by_side(left: str, right: str, gap: int) -> str:
    left_lines = left.strip("\n").splitlines()
    right_lines = right.strip("\n").splitlines()

    # colorize independently
    left_lines = _colorize_lines(left_lines, COLORS_WARM)
    right_lines = _colorize_lines(right_lines, COLORS_COOL)

    # compute sizes
    left_w = max(len(line) for line in left_lines) if left_lines else 0
    cols = shutil.get_terminal_size((120, 25)).columns
    # best effort: if too narrow, stack instead of side-by-side
    if cols < left_w + gap + 20:  # 20 ~ minimal title width
        return "\n".join([*left_lines, "", *right_lines]) + "\n"

    padded_left = _pad(left_lines, left_w)
    height = max(len(padded_left), len(right_lines))
    padded_left += [" " * left_w] * (height - len(padded_left))
    right_lines += [""] * (height - len(right_lines))

    rows = []
    spacer = " " * gap
    for L, R in zip(padded_left, right_lines, strict=False):
        rows.append(f"{L}{spacer}{R}")
    return "\n".join(rows) + "\n"

def print_banner() -> None:
    """
    Print the banner exactly once, only in the main process.
    Disable with WORP_BANNER=0 or NO_COLOR for monochrome.
    """
    if os.getenv(ENV_ENABLE, "1") in {"0", "false", "False", "no", "NO"}:
        return
    if getattr(print_banner, "_printed", False):
        return
    if current_process().name != "MainProcess":
        return
    print_banner._printed = True

    sys.stdout.write(_side_by_side(PLANET, TITLE, GAP))
    sys.stdout.flush()
