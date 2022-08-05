from napari_live_flim import FlimViewer
import time
import pytestqt
import inspect
from qtpy.QtWidgets import QWidget
from napari import Viewer
from napari.qt import QtViewer

# make_napari_viewer is a pytest fixture that returns a napari viewer object
# capsys is a pytest fixture that captures stdout and stderr output streams
def test_port_widget(make_napari_viewer, capsys, qtbot):
    # make viewer and add an image layer using our fixture
    viewer : Viewer = make_napari_viewer()

    # create our widget, passing in the viewer
    my_widget = FlimViewer(viewer)
    series_viewer = my_widget.series_viewer

    # call our widget method
    with qtbot.waitSignal(series_viewer.port_widget.port_line_edit.textChanged, raising=True):
        series_viewer.port_widget.port_line_edit.setText("5555")
    assert series_viewer.port == 5555

    series_viewer.tear_down()
