import logging
from dataclasses import dataclass
from numpy import ndarray
from qtpy.QtCore import QObject, Signal
from flimstream import Receiver, SeriesReceiver
from napari.qt.threading import thread_worker
from ._dataclasses import *

class FlimReceiver(QObject):
    new_series = Signal(SeriesMetadata)
    end_series = Signal()
    new_element = Signal(ElementData)

    receiver_worker = None

    def start_receiving(self, port):
        self.stop_receiving()
        print("creating new receiver on port", port)
        receiver = Receiver(port)
        self.receiver_worker = self.receive(receiver)
        self.receiver_worker.quit = lambda: receiver.quit()
        self.receiver_worker.yielded.connect(lambda element: self.new_element.emit(element))
        self.receiver_worker.start()
        
    def stop_receiving(self):
        if self.receiver_worker is not None:
            print("quitting receiver...")
            self.receiver_worker.quit()
            self.receiver_worker = None

    @thread_worker
    def receive(self, receiver : Receiver):
        series_no = -1
        while True:
            series_receiver = receiver.wait_and_receive_series()
            if not series_receiver or series_receiver is Ellipsis:
                logging.info("Worker exiting")
                return
            series_no += 1
            self.new_series.emit(SeriesMetadata(series_no, receiver.port, series_receiver.shape))
            while True:
                seqno, frame = series_receiver.wait_and_receive_element()
                if frame is None:
                    self.end_series.emit()
                    break
                if frame is Ellipsis:
                    logging.info("Worker exiting")
                    return
                yield ElementData(series_no, seqno, frame)