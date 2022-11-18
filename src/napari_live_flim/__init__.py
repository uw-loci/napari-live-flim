from ._controller import Controller
from napari import Viewer
from qtpy.QtWidgets import QWidget, QFormLayout

from ._version import __version__


class FlimViewer(QWidget):
    """
    A QWidget that encapsulates the napari plugin

    In order to safely run this plugin, Napari should be run as shown here:
    https://forum.image.sc/t/multiple-viewer-in-one-napari-window-example/69627/7?u=facetorched
    """

    _instance = None
    # QWidget.__init__ can optionally request the napari viewer instance
    def __init__(self, napari_viewer : Viewer):
        super().__init__()
        if FlimViewer.instance() is not None:
            self.series_viewer = FlimViewer.instance().series_viewer
        else:
            self.series_viewer = Controller(napari_viewer)

        self.layout = QFormLayout()
        self.setLayout(self.layout)
        self.layout.addRow(self.series_viewer.port_widget.group)
        self.layout.addRow(self.series_viewer.flim_params_widget.group)
        self.layout.addRow(self.series_viewer.display_settings_widget.group)
        self.layout.addRow(self.series_viewer.actions_widget.group)
        self.layout.addRow(self.series_viewer.save_settings_widget.group)

        FlimViewer._instance = self # singleton reference

    @classmethod
    def instance(cls):
        # return current/last widget instance (or None).
        return cls._instance
