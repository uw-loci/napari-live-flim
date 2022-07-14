# Here's a script to play around with napari to help with development

from magicgui import magicgui
import napari
import numpy as np

from napari_live_flim.data_generator import data_generator, SHAPE
from napari_live_flim._constants import *


if __name__ == "__main__":
    viewer = napari.Viewer()
    dgen = data_generator()
    img = np.array([next(dgen), next(dgen), next(dgen)]).reshape((SHAPE) + (3,))
    img = img[:3]
    print(img.shape)
    image = viewer.add_image(img, rgb=True)
    @magicgui(call_button="run")
    def call_button():
        print(image.rgb)
    viewer.window.add_dock_widget(call_button, area="left")
    napari.run()
