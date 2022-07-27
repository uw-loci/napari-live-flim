# Here's a script to play around with napari to help with development

from magicgui import magicgui
import napari
from napari.qt import QtViewer
from napari.components.viewer_model import ViewerModel
import vispy.color
from skimage import data
import numpy as np
import colorsys
from qtpy.QtWidgets import QMainWindow, QWidget
from qtpy.QtCore import Qt
from napari.utils.colormaps.colormap_utils import AVAILABLE_COLORMAPS, ensure_colormap, Colormap

from napari_live_flim.data_generator import data_generator, SHAPE
from napari_live_flim._constants import *

IMAGE_SIZE = 20

if __name__ == "__main__":
    viewer = napari.Viewer()
    co_viewer_model = ViewerModel(title="Phasor Viewer")
    
    co_viewer = QtViewer(co_viewer_model)
    co_viewer_model.add_image(np.eye(10))
    
    viewer.window.add_dock_widget(co_viewer)
    viewer.window.add_dock_widget(co_viewer.dockLayerList)

    viewer.add_image(data.astronaut(), scale=[-1, 1])
    viewer.text_overlay.visible = True
    viewer.text_overlay.text = "crungog"

    @magicgui(call_button="run")
    def call_button():
        print(AVAILABLE_COLORMAPS)
    viewer.window.add_dock_widget(call_button, area="left")
    napari.run()
