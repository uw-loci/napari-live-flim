import numpy as np
from napari_live_flim._series_viewer import LifetimeImageProxy, ComputeTask

NUM_PIXELS = 128
RGB_IMAGE_SHAPE = (NUM_PIXELS, NUM_PIXELS, 3)

def test_slicing():
    
    indices = (slice(None, None, None), slice(None, None, None), 10)
    lip = LifetimeImageProxy([ComputeTask(0, None)], RGB_IMAGE_SHAPE, np.zeros(RGB_IMAGE_SHAPE))
    assert lip[indices].shape == (1, NUM_PIXELS, 3)

if __name__ == "__main__":
    test_slicing()