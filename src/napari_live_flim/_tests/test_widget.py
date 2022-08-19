from napari_live_flim import FlimViewer
from napari import Viewer
from pytestqt.plugin import QtBot
from napari_live_flim._constants import *

# make_napari_viewer is a pytest fixture that returns a napari viewer object
# capsys is a pytest fixture that captures stdout and stderr output streams
def test_change_settings(make_napari_viewer, capsys, qtbot : QtBot):
    # make viewer and add an image layer using our fixture
    viewer : Viewer = make_napari_viewer()

    # create our plugin widget, passing in the viewer
    my_widget = FlimViewer(viewer)
    series_viewer = my_widget.series_viewer

    # test port widget
    with qtbot.waitSignal(series_viewer.port_widget.port_line_edit.textChanged, raising=True):
        series_viewer.port_widget.port_line_edit.setText("5555")
    assert series_viewer.port == 5555

    # test flim params widget
    test_params = FlimParams(period=3.14, fit_start=1, fit_end=42)
    with qtbot.waitSignal(series_viewer.flim_params_widget.changed, raising=True):
        series_viewer.flim_params_widget.period.setValue(test_params.period)
        series_viewer.flim_params_widget.fit_end.setValue(test_params.fit_end)
        series_viewer.flim_params_widget.fit_start.setValue(test_params.fit_start)
    assert series_viewer.flim_params == test_params

    # test display settings widget
    test_settings = DisplaySettings(max_chisq=86, min_tau=2.72, max_tau=49, colormap="intensity")
    with qtbot.waitSignal(series_viewer.display_settings_widget.changed, raising=True):
        series_viewer.display_settings_widget.max_chisq.setValue(test_settings.max_chisq)
        series_viewer.display_settings_widget.min_tau.setValue(test_settings.min_tau)
        series_viewer.display_settings_widget.max_tau.setValue(test_settings.max_tau)
        series_viewer.display_settings_widget.colormap.setCurrentText(test_settings.colormap)
    assert series_viewer.display_settings == test_settings

    # test delta snapshots
    with qtbot.waitSignal(series_viewer.actions_widget.delta_snapshots.toggled, raising=True):
        series_viewer.actions_widget.delta_snapshots.setChecked(True)
    assert series_viewer.delta_snapshots == True

    series_viewer.tear_down()
