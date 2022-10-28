import numpy as np

SAMPLES = 256
A_IN = 10.0
TAU_IN = 1.0
PERIOD = 0.04

SHAPE = (SAMPLES, SAMPLES, SAMPLES)
A_INC = 10/SAMPLES
TAU_INC = 1/SAMPLES

def data_generator(n):
    """
    generate `n` noisy FLIM image frames that vary horizontally with lifetime
    and vertically with aplitude. Generated frames are accumulated.
    """
    time = np.linspace(0, (SAMPLES - 1) * PERIOD, SAMPLES, dtype=np.float32)
    photon_count = np.empty(SHAPE, dtype=np.uint16)
    for _ in range(n):
        for i in range(SAMPLES):
            for j in range(SAMPLES):
                # lifetime varies horizontally
                tau = TAU_IN + j * TAU_INC
                # intensity varies vertically. dividing by tau keeps the 
                # intensity (integral of the exponential) constant horizontally
                a = (A_IN + i * A_INC) * (TAU_IN / tau)
                decay = a * np.exp(-time / tau) * np.random.poisson(size=SAMPLES)
                decay[decay < 0] = 0
                photon_count[i][j] += np.round(decay).astype(np.uint16)
        yield photon_count.copy()
