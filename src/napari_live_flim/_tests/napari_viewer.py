# Here's a script to play around with napari to help with development

from magicgui import magicgui
import napari
import vispy.color
import numpy as np
import colorsys
from napari.utils.colormaps.colormap_utils import AVAILABLE_COLORMAPS, ensure_colormap, Colormap

from napari_live_flim.data_generator import data_generator, SHAPE
from napari_live_flim._constants import *

IMAGE_SIZE = 20

if __name__ == "__main__":
    viewer = napari.Viewer()
    greg = Colormap([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], "greg")
    img = np.asarray([colorsys.hsv_to_rgb(hue, 1.0, 1.0) for hue in np.linspace(0,1,IMAGE_SIZE)])
    img = np.broadcast_to(img, (IMAGE_SIZE,) + img.shape)
    image = viewer.add_image(img)
    img = np.asarray([(1.0, 1.0, 1.0, a) for a in np.linspace(0,1,IMAGE_SIZE)])
    img = np.broadcast_to(img, (IMAGE_SIZE,) + img.shape).swapaxes(0,1)
    image = viewer.add_image(img)
    ensure_colormap(greg)
    @magicgui(call_button="run")
    def call_button():
        print(AVAILABLE_COLORMAPS)
    viewer.window.add_dock_widget(call_button, area="left")
    napari.run()
