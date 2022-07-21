import numpy as np
import colorsys
from pathlib import Path
from matplotlib.colors import ListedColormap, Normalize

OPTIONS_VERSION = 1
MAX_VALUE = 1000000 # large number used as max value for certain user inputs

FONT_SIZE = 10
PHASOR_SCALE = 1000
PHASOR_OPACITY_FACTOR = 0.2
DEFAULT_PERIOD = 0.04
DEFUALT_MIN_INTENSITY = 10
DEFUALT_MAX_CHISQ = float(MAX_VALUE)
DEFAULT_MAX_TAU = 10.0
COLOR_DEPTH = 256

# not actually empty. Ideally I could use None as input to napari but it doesn't like it
EMPTY_RGB_IMAGE = np.zeros((1,1,3))
EMPTY_PHASOR_IMAGE = np.zeros((1,2))
EMPTY_PHOTON_COUNT = np.zeros((1, 1, 0), dtype=np.float32)

DEFUALT_LIFETIME_SELECTION = np.array([[0, 0], [0, 20], [40, 20], [40, 0]])
DEFUALT_PHASOR_SELECTION = np.array([
    [PHASOR_SCALE//2, PHASOR_SCALE//2], 
    [PHASOR_SCALE//2, PHASOR_SCALE*2//3],
    [PHASOR_SCALE*2//3, PHASOR_SCALE*2//3],
    [PHASOR_SCALE*2//3, PHASOR_SCALE//2]
])

COLOR_DICT = {  "red":"#FF0000",
                "green":"#00FF00",
                "blue":"#0000FF",
                "cyan":"#00FFFF",
                "magenta":"#FF00FF",
                "yellow":"#FFFF00",
            }

COLORMAPS = {"intensity" : ListedColormap([1,1,1], "intensity")}

for path in Path(__file__).resolve().parent.joinpath('colormaps').iterdir():
    arr = np.genfromtxt(path, delimiter=",", dtype=int)
    norm = Normalize(0, 255)
    name = path.stem
    cm = ListedColormap(norm(arr), name)
    COLORMAPS[name] = cm