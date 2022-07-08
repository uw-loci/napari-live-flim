import subprocess
import napari
from napari_live_flim import FlimViewer
from napari_live_flim._dataclasses import *
import numpy as np
import flimstream
import time
import logging
from magicgui import magicgui
from napari.qt.threading import thread_worker
import pathlib
import copy

SAMPLES = 128
A_IN = 10.0
TAU_IN = 1.0
PERIOD = 0.04
PORT = 5555

SHAPE = (SAMPLES, SAMPLES, SAMPLES)
A_INC = 10/SAMPLES
TAU_INC = 1/SAMPLES

series_no = -1

def data_generator():
    time = np.linspace(0, (SAMPLES - 1) * PERIOD, SAMPLES, dtype=np.float32)
    photon_count = np.empty(SHAPE, dtype=np.uint16)
    while True:
        for i in range(SAMPLES):
            for j in range(SAMPLES):
                a = A_IN + i * A_INC
                tau = TAU_IN + j * TAU_INC
                decay = a * np.exp(-time / tau) * np.random.poisson(size=SAMPLES)
                decay[decay < 0] = 0
                photon_count[i][j] += decay.astype(np.uint16)
                if(i == j == 0):
                    print(photon_count[i][j])
        yield photon_count

@thread_worker
def send_series(viewer : FlimViewer, series_no, frames, interval):
    data_gen = data_generator()
    viewer.new_series(SeriesMetadata(series_no, "TEST", SHAPE))

    for i in range(frames):
        start = time.time()

        frame = next(data_gen)
        viewer.new_element(ElementData(series_no, i, frame))
        finish = time.time()
        excess = finish - start - interval
        if excess > 0:
            logging.warning(f"Frame interval exceeded by {excess} s")
        else:
            time.sleep(-excess)
    viewer.end_series()

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
    flim_viewer.port_widget.port_line_edit.setText(str(PORT))
    create_send_widget(viewer, flim_viewer)
    napari.run()