from ._series_viewer import SeriesViewer
from napari import Viewer
from PyQt5 import QtGui
from qtpy import QtCore
from qtpy.QtWidgets import QWidget, QFormLayout, QApplication

# https://forum.image.sc/t/multiple-viewer-in-one-napari-window-example/69627/7?u=facetorched
# remove this when the fix is added to napari
QApplication.setAttribute(QtCore.Qt.AA_ShareOpenGLContexts)

__version__ = "0.0.1"

class FlimViewer(QWidget):
    # QWidget.__init__ can optionally request the napari viewer instance
    def __init__(self, napari_viewer : Viewer):
        super().__init__()

        self.series_viewer = SeriesViewer(napari_viewer)
        self.destroyed.connect(self.series_viewer.tear_down)

        self.layout = QFormLayout()
        self.setLayout(self.layout)
        self.layout.addRow(self.series_viewer.port_widget.group)
        self.layout.addRow(self.series_viewer.flim_params_widget.group)
        self.layout.addRow(self.series_viewer.display_settings_widget.group)
        self.layout.addRow(self.series_viewer.actions_widget.group)
        self.layout.addRow(self.series_viewer.save_settings_widget.group)

    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:
        super().closeEvent(a0)
        print(type(a0))

    def hideEvent(self, a0: QtGui.QHideEvent) -> None:
        super().hideEvent(a0)
        print(type(a0))