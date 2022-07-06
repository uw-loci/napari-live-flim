from napari_live_flim import FlimViewer

# make_napari_viewer is a pytest fixture that returns a napari viewer object
# capsys is a pytest fixture that captures stdout and stderr output streams
def test_port_widget(make_napari_viewer, capsys):
    # make viewer and add an image layer using our fixture
    viewer = make_napari_viewer()

    # create our widget, passing in the viewer
    my_widget = FlimViewer(viewer)

    # call our widget method
    my_widget.port_widget.port_line_edit.setText("5555")

    # read captured output and check that it's as we expected
    captured = capsys.readouterr()
    assert captured.out == "creating new receiver on port 5555\n"

    my_widget.phasor_viewer.close_all()
