import numpy as np

SAMPLES = 128
A_IN = 10.0
TAU_IN = 1.0
PERIOD = 0.04

SHAPE = (SAMPLES, SAMPLES, SAMPLES)
A_INC = 10/SAMPLES
TAU_INC = 1/SAMPLES

def data_generator():
    """
    generate noisy FLIM data that varies horizontally with lifetime
    and vertically with aplitude. Generated frames are accumulated.
    """
    time = np.linspace(0, (SAMPLES - 1) * PERIOD, SAMPLES, dtype=np.float32)
    photon_count = np.empty(SHAPE, dtype=np.uint16)
    while True:
        for i in range(SAMPLES):
            for j in range(SAMPLES):
                a = A_IN + i * A_INC
                tau = TAU_IN + j * TAU_INC
                decay = a * np.exp(-time / tau) * np.random.poisson(size=SAMPLES)
                decay[decay < 0] = 0
                photon_count[i][j] += np.round(decay).astype(np.uint16)
        yield photon_count
