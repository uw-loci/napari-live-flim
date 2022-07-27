# simulate running the application and performing acquisiton
# for testing the GUI and for demos

import napari
from napari_live_flim import FlimViewer
from napari_live_flim._dataclasses import *
import time
import logging
from magicgui import magicgui
from napari.qt.threading import thread_worker
from napari_live_flim.data_generator import data_generator, SHAPE

logging.basicConfig(level=logging.INFO)

PORT = 5555
series_no = -1

@thread_worker
def send_series(viewer : FlimViewer, series_no, frames, interval):
    data_gen = data_generator()
    viewer.series_viewer.new_series(SeriesMetadata(series_no, "TEST", SHAPE))

    for i in range(frames):
        start = time.time()

        frame = next(data_gen)
        viewer.series_viewer.new_element(ElementData(series_no, i, frame))
        finish = time.time()
        excess = finish - start - interval
        if excess > 0:
            logging.warning(f"Frame interval exceeded by {excess} s")
        else:
            time.sleep(-excess)
    viewer.series_viewer.end_series()

def create_send_widget(viewer : napari.Viewer, flim_viewer : FlimViewer):
    @magicgui(call_button="send",)
    def send_widget(frames=10, interval=0.5):
        global series_no
        series_no += 1
        worker = send_series(flim_viewer, series_no, frames, interval)
        worker.start()
    viewer.window.add_dock_widget(send_widget, area="bottom")

if __name__ == "__main__":
    viewer = napari.Viewer()
    flim_viewer = FlimViewer(viewer)
    viewer.window.add_dock_widget(flim_viewer)
    flim_viewer.series_viewer.port_widget.port_line_edit.setText(str(PORT))
    create_send_widget(viewer, flim_viewer)
    napari.run()