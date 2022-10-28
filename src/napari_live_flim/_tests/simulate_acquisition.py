# simulate running the application and performing acquisiton
# for testing the GUI and for demos

import napari
from napari_live_flim import FlimViewer
import time
import logging
from magicgui import magicgui
from napari.qt.threading import thread_worker
from napari_live_flim.data_generator import data_generator, SHAPE
from napari_live_flim._constants import *
from napari_live_flim._dataclasses import *
from napari_live_flim._flim_receiver import SeriesSender
from qtpy import QtWidgets, QtCore

logging.basicConfig(level=logging.INFO)

series_no = -1

@thread_worker
def send_series_direct(viewer : FlimViewer, series_no, frames, interval):
    data_gen = data_generator(frames)
    frames = list(data_gen) # generating on each loop causes too much delay
    viewer.series_viewer.new_series(SeriesMetadata(series_no, "TEST", SHAPE))
    for i, frame in enumerate(frames):
        start = time.time()
        viewer.series_viewer.new_element(ElementData(series_no, i, frame))
        finish = time.time()
        excess = finish - start - interval
        if excess > 0:
            logging.warning(f"Frame interval exceeded by {excess} s")
        else:
            time.sleep(-excess)
    viewer.series_viewer.end_series()

def create_send_widget(viewer : napari.Viewer, flim_viewer : FlimViewer):
    @magicgui(call_button="send direct", send_udp = {"widget_type" : "PushButton", "label" : "send via UDP"})
    def send_widget(send_udp=False, frames=10, interval=0.5):
        global series_no
        series_no += 1
        worker = send_series_direct(flim_viewer, series_no, frames, interval)
        worker.start()
    viewer.window.add_dock_widget(send_widget, area="bottom")

    def send_udp_callback():
        frames = send_widget.frames.value
        interval = send_widget.interval.value
        worker = send_series_udp(DEFAULT_PORT, frames, interval)
        worker.start()
            

    send_widget.send_udp.clicked.connect(send_udp_callback)

@thread_worker
def send_series_udp(port, frames, interval):
    data_gen = data_generator(frames)
    frames = list(data_gen)
    sender = SeriesSender(np.dtype(np.uint16), SHAPE, port)
    sender.start()
    for i, frame in enumerate(frames):
        start = time.time()
        sender.send_element(i, frame)
        finish = time.time()
        excess = finish - start - interval
        if excess > 0:
            logging.warning(f"Frame interval exceeded by {excess} s")
        else:
            time.sleep(-excess)
    sender.end()

if __name__ == "__main__":
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_ShareOpenGLContexts)
    viewer = napari.Viewer()
    flim_viewer = FlimViewer(viewer)
    viewer.window.add_dock_widget(flim_viewer)
    create_send_widget(viewer, flim_viewer)
    napari.run()