import numpy as np
import colorsys

FONT_SIZE = 10
PHASOR_SCALE = 1000
PHASOR_OPACITY_FACTOR = 0.2
DEFAULT_PERIOD = 0.04
DEFUALT_MIN_INTENSITY = 10
DEFUALT_MAX_CHISQ = 200.0
DEFAULT_MAX_TAU = 10.0
COLOR_DEPTH = 256
COLORMAP = np.array([colorsys.hsv_to_rgb(f, 1.0, 1) for f in np.linspace(0,1,COLOR_DEPTH)], dtype=np.float32)

OPTIONS_VERSION = 1
MAX_VALUE = 1000000 # large number used as max value for certain user inputs

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