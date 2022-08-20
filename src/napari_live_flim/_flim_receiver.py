import logging
from qtpy.QtCore import QObject, Signal
from napari.qt.threading import thread_worker
from ._constants import *
from ._dataclasses import *
import numpy as np
import os.path
import socket
import mmap
import tempfile

class FlimReceiver(QObject):
    """
    A class that uses a thread worker receive FLIM data via the `Receiver`.
    emits signals for `new_series`, `end_series` and `new_element` to be connected to
    on the main thread.
    """

    new_series = Signal(SeriesMetadata)
    end_series = Signal()
    new_element = Signal(ElementData)

    receiver_worker = None

    def start_receiving(self, port):
        self.stop_receiving()
        logging.info(f"Creating new receiver on port {port}")
        receiver = Receiver(port)
        self.receiver_worker = self.receive(receiver)
        self.receiver_worker.quit = lambda: receiver.quit()
        self.receiver_worker.yielded.connect(lambda element: self.new_element.emit(element))
        self.receiver_worker.start()
        
    def stop_receiving(self):
        if self.receiver_worker is not None:
            logging.info("Quitting receiver...")
            self.receiver_worker.quit()
            self.receiver_worker = None

    @thread_worker
    def receive(self, receiver : "Receiver"):
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
                # for now, we ignore all but the first channel
                if frame.ndim > 3:
                    frame = frame[tuple([0] * (frame.ndim - 3))]
                yield ElementData(series_no, seqno, frame)

class Receiver:
    def __init__(self, port, addr=None):
        self.port = port
        self.addr = (addr if addr else "127.0.0.1")
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind((self.addr, self.port))

    def wait_and_receive_series(self):
        """Returns a SeriesReceiver when a new incoming series is started.

        Returns Ellipsis when the receiver is quit.
        """

        while True:
            m = self._recvmsg()
            if isinstance(m, _NewSeriesMessage):
                return SeriesReceiver(self, m.dtype, m.shape, m.dirpath)
            if isinstance(m, _QuitMessage):
                self.socket.close()
                return Ellipsis
            # Ignore messages until start of new series

    def quit(self):
        """Quit the receiver and cause any waiting functions to return."""

        ssock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        ssock.sendto("quit".encode(), ("127.0.0.1", self.port))
        ssock.close()
        logging.info("Sent quit message")

    def _recvmsg(self):
        msg, addr = self.socket.recvfrom(512)
        msgstr = msg.decode()
        logging.info(f"Received message: {msgstr}")
        return _parse_message(msg)


class SeriesReceiver:
    def __init__(self, receiver, dtype, shape, dirpath):
        self.receiver = receiver
        self.dtype = dtype
        self.shape = shape
        self.dirpath = dirpath
        self.ended = False

    def wait_and_receive_element(self):
        """Return (seqno, array) when a data element is received.

        Returns (-1, None) when the end of a series is reached.
        Returns (-1, Ellipsis) when the receiver was quit.
        """

        if self.ended:
            return -1, None
        m = self.receiver._recvmsg()
        if isinstance(m, _SeriesElementMessage):
            array = _map_array(self.dtype, self.shape, self.dirpath, m.seqno)
            return m.seqno, array
        if isinstance(m, _EndSeriesMessage):
            self.ended = True
            return -1, None
        if isinstance(m, _QuitMessage):
            self.receiver.socket.close()
            return -1, Ellipsis
        raise IOError



class _NewSeriesMessage:
    def __init__(self, dtype, shape, dirpath):
        self.dtype = dtype
        self.shape = shape
        self.dirpath = dirpath


class _SeriesElementMessage:
    def __init__(self, seqno):
        self.seqno = seqno


class _EndSeriesMessage:
    pass


class _QuitMessage:
    pass


def _parse_message(msg):
    fields = msg.decode().split('\t')
    cmd = fields.pop(0)
    if cmd == "new_series":
        return _parse_message_new_series(fields)
    elif cmd == "element":
        return _parse_message_element(fields)
    elif cmd == "end_series":
        return _parse_message_end_series(fields)
    elif cmd == "quit":
        return _parse_message_quit(fields)
    else:
        logging.error(f"unknown message command: {cmd}")
        raise IOError


def _parse_message_new_series(fields):
    dtype = _parse_dtype(fields.pop(0))
    ndim = int(fields.pop(0))
    shape = tuple(int(fields.pop(0)) for i in range(ndim))
    dirpath = fields.pop(0)
    if len(fields):
        raise IOError
    return _NewSeriesMessage(dtype, shape, dirpath)


def _parse_message_element(fields):
    if len(fields) < 1:
        raise IOError
    return _SeriesElementMessage(int(fields[0]))


def _parse_message_end_series(fields):
    return _EndSeriesMessage()


def _parse_message_quit(fields):
    return _QuitMessage()


def _parse_dtype(d):
    if d == "u16":
        return np.uint16
    elif d == "u8":
        return np.uint8
    else:
        raise IOError

def _map_array(dtype, shape, dirpath, index):
    path = os.path.join(dirpath, str(index))
    arr = np.memmap(path, dtype=dtype, shape=shape, mode="c")
    logging.info(f"mapped {dtype} array {shape} at {path}")
    return arr

class SeriesSender:
    """
    A class to send a series via UDP and temporary memory mapped files to be recieved by the `Receiver`
    Intended to be used for testing
    """
    def __init__(self, dtype, element_shape, port, addr=None, dirpath=None):
        self.dtype = dtype
        self.element_shape = element_shape
        self.port = port
        self.addr = (addr if addr else "127.0.0.1")

        self.tempdir = tempfile.TemporaryDirectory(dir=dirpath)
        self.dirpath = self.tempdir.name

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def _send_message(self, msgbytes):
        self.socket.sendto(msgbytes, (self.addr, self.port))
        msgstr = msgbytes.decode()
        logging.info(f"Sent message: {msgstr}")

    def _write_array(self, dirpath, seqno, dtype, arr):
        path = os.path.join(dirpath, str(seqno))
        size = arr.size * dtype.itemsize
        with open(path, "wb+") as f:
            with mmap.mmap(f.fileno(), size) as m:
                b = np.frombuffer(m, dtype=dtype).reshape(arr.shape)
                b[:] = arr
                del b
        logging.info(f"Wrote {dtype} array {arr.shape} to {path}")

    def start(self):
        dt = None
        if self.dtype == np.uint16:
            dt = "u16"
        elif self.dtype == np.uint8:
            dt = "u8"
        assert dt

        ndim = len(self.element_shape)
        shape = "\t".join(str(d) for d in self.element_shape)
        dirpath = self.dirpath

        self._send_message(f"new_series\t{dt}\t{ndim}\t{shape}\t{dirpath}".encode())

    def send_element(self, seqno, element):
        assert element.shape == self.element_shape
        self._write_array(self.dirpath, seqno, self.dtype, element)
        self._send_message(f"element\t{seqno}".encode())

    def end(self):
        self._send_message("end_series".encode())