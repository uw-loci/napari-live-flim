import time
from concurrent.futures import Future
from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import flimlib
import logging
import numpy as np
from napari.layers import Image
from qtpy.QtCore import QObject, Signal

from scipy.spatial import KDTree
from superqt import ensure_main_thread

if TYPE_CHECKING:
    from ._controller import Controller

from ._constants import *
from ._dataclasses import *
from .gather_futures import gather_futures
from .timing import timing

_receive_times = {}
_compute_times = []

class ComputeTask:
    def __init__(self, step : int, series_viewer : "SeriesViewer"):
        self._valid = True
        self._step = step
        self._series_viewer = series_viewer # need this to retrieve the most recent photon_count/params

        self.intensity = None
        self.lifetime_image = None
        self.phasor = None
        self.phasor_quadtree = None
        self.phasor_image = None
        self.phasor_face_color = None
        self.done = None

    def start(self):
        if self._series_viewer is not None and not self.all_started(): # dont start more than once
            photon_count = self._series_viewer.get_photon_count(self._step)
            params = self._series_viewer.settings.flim_params
            display_settings = self._series_viewer.settings.display_settings

            self.intensity = EXECUTOR.submit(compute_intensity, photon_count)
            self.lifetime_image = EXECUTOR.submit(compute_lifetime_image, photon_count, self.intensity, params, display_settings)
            self.phasor = EXECUTOR.submit(compute_phasor, photon_count, params)
            self.phasor_quadtree = EXECUTOR.submit(compute_phasor_quadtree, self.phasor)
            self.phasor_image = EXECUTOR.submit(compute_phasor_image, self.phasor)
            self.phasor_face_color = EXECUTOR.submit(compute_phasor_face_color, self.intensity)
            self.done = gather_futures(self.intensity, self.lifetime_image, self.phasor, self.phasor_quadtree, self.phasor_image, self.phasor_face_color)
            self.done.add_done_callback(self._stop_benchmark)
            self.done.add_done_callback(self._series_viewer.compute_done_callback)

    def cancel(self):
        if self.all_started(): # if user is looking at a snapshot, latest may not have even been started
            self.intensity.cancel()
            self.lifetime_image.cancel()
            self.phasor.cancel()
            self.phasor_quadtree.cancel()
            self.phasor_image.cancel()
            self.phasor_face_color.cancel()
            self.done.cancel()
    
    def all_started(self):
        return self.done is not None

    def is_running(self):
        return self.all_started() and self.done.running()

    def all_done(self):
        return self.all_started() and self.done.done()

    def invalidate(self):
        if self.all_started():
            self._valid = False

    def is_valid(self):
        return self._valid
    
    def _stop_benchmark(self, done):
        global _compute_times, _receive_times
        
        fn = self._series_viewer.get_frame_no(self._step)
        if fn in _receive_times.keys():
            t = (time.perf_counter() - _receive_times.pop(fn)) * 1000
            _compute_times += [t]
            logging.info(f"Processing frame {fn} took {t} milliseconds. Median {np.median(_compute_times)}")
        else:
            logging.error(f"Benchmarking failed! frame number {fn} was not found")
        
        logging.info(f"There are {len(_receive_times)} frames that were skipped or not yet processed")

class LifetimeImageProxy:
    """
    An array-like object backed by the collection of lifetime_image tasks
    """

    def __init__(self, tasks_list : List[ComputeTask], image_shape, old_proxy : np.ndarray, dtype=np.float32):
        if not len(tasks_list):
            # At least for now, dont allow empty
            raise ValueError("LifetimeImageProxy must have at least one task")
        self._arrays = tasks_list
        self._image_shape = image_shape
        if old_proxy.shape != image_shape:
            raise ValueError(f"old proxy with shape {old_proxy.shape} must match new shape {image_shape}")
        self._most_recent = old_proxy
        self._dtype = dtype

    def get_old_proxy(self):
        return self._most_recent

    @property
    def ndim(self):
        return 1 + len(self._image_shape)

    @property
    def dtype(self):
        return self._dtype

    @property
    def shape(self):
        return (len(self._arrays),) + self._image_shape

    def __getitem__(self, slices):
        if not isinstance(slices, tuple):
            slices = (slices,)
        ndslices = slices[1:]
        s0 = slices[0]
        if isinstance(s0, slice):
            start, stop, step = s0.indices(len(self._arrays))
            dim0 = (stop - start) // step
            shape = (dim0,) + get_sliced_shape(self._image_shape, ndslices)
            ret = np.empty(shape, dtype=self._dtype)
            j0 = 0
            for i0 in range(start, stop, step):
                ret[j0] = self._get_slices_at_index(i0, ndslices)
                j0 += 1
            return ret
        else:  # s0 is an integer
            return self._get_slices_at_index(s0, ndslices)

    def _get_slices_at_index(self, index, slices):
        task = self._arrays[index]
        if task.all_done():
            self._most_recent = task.lifetime_image.result(timeout=0)
        else:
            task.start()
        return self._most_recent[slices]

    def __array__(self):
        logging.warning("Full array of proxy array has been unexpectedly requested.")
        return self[:]
    
    def __len__(self):
        return np.prod(self.shape)

def create_lifetime_image_proxy(series_viewer : "SeriesViewer", dtype=np.float32):
    tasks_list = series_viewer.get_tasks_list()

    image_shape = series_viewer.get_image_shape() + (4,) # rgb
    possible_old_proxy = series_viewer.lifetime_image.data
    if isinstance(possible_old_proxy, LifetimeImageProxy):
        old_proxy = possible_old_proxy.get_old_proxy()
    else:
        old_proxy = np.zeros(image_shape, dtype=dtype)
    
    return LifetimeImageProxy(tasks_list, image_shape, old_proxy, dtype=dtype)

def get_sliced_shape(shape : tuple, slices):
    """
    predicts the resulting shape of an array-like with 
    the given `shape` after indexing with `slices`
    """
    ret = ()
    for i, dim in enumerate(shape):
        if i < len(slices):
            s = slices[i]
            if isinstance(s, slice):
                start, stop, step = s.indices(dim)
                ret += ((stop - start) // step,)
            elif not np.isscalar(s):
                raise TypeError(f"slices must contain either slice or scalar, not {type(s)}")
        else:
            ret += (dim,)
    return ret

# about 0.1 seconds for 256x256x256 data
@timing
def compute_lifetime_image(photon_count : np.ndarray, intensity_future : Future, params : FlimParams, display_settings : DisplaySettings):
    period = params.period
    fstart = params.fit_start if params.fit_start < photon_count.shape[-1] else photon_count.shape[-1]
    fend =  params.fit_end if params.fit_end <= photon_count.shape[-1] else photon_count.shape[-1]
    # computing chi squared is a significant portion of the compute time
    rld = flimlib.GCI_triple_integral_fitting_engine(period, photon_count, fit_start=fstart, fit_end=fend, compute_fitted=False, compute_residuals=False)
    tau = rld.tau

    intensity = intensity_future.result()
    invalid_indexer = np.where(
        (np.isnan(tau)) |
        (rld.chisq > display_settings.max_chisq) |
        (tau < display_settings.min_tau) |
        (tau > display_settings.max_tau)
    )
    intensity[invalid_indexer] = np.nan
    tau[invalid_indexer] = np.nan
    intensity = normalize(intensity)
    np.nan_to_num(intensity, copy=False)
    tau = normalize(tau, min_in=display_settings.min_tau, max_in=display_settings.max_tau)
    np.nan_to_num(tau, copy=False)
    rgb_tau = COLORMAPS[display_settings.colormap](tau)
    rgb_tau[...,:3] *= intensity[..., np.newaxis]
    return rgb_tau

def compute_intensity(photon_count : np.ndarray) -> np.ndarray:
    return np.nansum(photon_count, axis=-1)

def compute_phasor_image(phasor_future : Future):
    phasor = phasor_future.result()
    return phasor.reshape(-1,phasor.shape[-1])

def compute_phasor_face_color(intensity_future : Future):
    intensity = normalize(intensity_future.result())
    phasor_intensity = intensity.ravel() * PHASOR_OPACITY_FACTOR
    color = np.broadcast_to(1.0, phasor_intensity.shape)
    return np.asarray([color,color,color,phasor_intensity]).T

def normalize(data : np.ndarray, min_in=None, max_in=None, min_out=0, max_out=1) -> np.ndarray:
    if min_in is None:
        min_in = np.nanmin(data)
    if max_in is None:
        max_in = np.nanmax(data)
    scale_in = max_in - min_in
    if scale_in <= 0 or np.isnan(scale_in):
        return(np.zeros_like(data) + min_out)
    scale_out = max_out - min_out
    return ((data - min_in) * (scale_out / scale_in)) + min_out

@timing
def compute_phasor(photon_count : np.ndarray, params : FlimParams):
    period = params.period
    fstart = params.fit_start if params.fit_start < photon_count.shape[-1] else photon_count.shape[-1]
    fend =  params.fit_end if params.fit_end <= photon_count.shape[-1] else photon_count.shape[-1]
    # about 0.5 sec for 256x256x256 data
    phasor = flimlib.GCI_Phasor(period, photon_count, fit_start=fstart, fit_end=fend, compute_fitted=False, compute_residuals=False, compute_chisq=False)
    #reshape to work well with mapping / creating the phasor plot. Result has shape (height, width, 2)
    return np.dstack([phasor.v, phasor.u])

def compute_phasor_quadtree(phasor_future : Future):
    phasor = phasor_future.result() * PHASOR_SCALE
    # workaround that fixes https://github.com/scipy/scipy/issues/14527
    np.nan_to_num(phasor, copy=False, nan=np.inf)
    quadtree = KDTree(phasor.reshape(-1, phasor.shape[-1]))
    return quadtree

@dataclass
class _SnapshotData:
    element : ElementData
    tasks : ComputeTask

class SeriesViewer(QObject):
    compute_done = Signal()

    def __init__(self, image : Image, shape, settings : Settings):
        super(SeriesViewer, self).__init__()

        self._snapshots : List[_SnapshotData]= []
        self.shape = shape
        self.lifetime_image = image
        self.settings = settings

    def validate_tasks(self, index):
        """
        Set the task at the given index to a new task if it is not valid
        """
        index %= len(self._snapshots) # convert to positive index
        tasks = self._snapshots[index].tasks
        if tasks is None:
            self._snapshots[index].tasks = ComputeTask(index, self)
        elif not tasks.is_valid() and not tasks.is_running():
            self._snapshots[index].tasks = ComputeTask(index, self)
            tasks.cancel()

    def snap(self):
        """
        Create a new snapshot and assigning it compute tasks. Creates
        new compute tasks only if necessary.
        """
        if self.has_data():
            prev = self._snapshots[-1]
            if self.settings.delta_snapshots:
                self._snapshots += [_SnapshotData(prev.element, None)]
                self.validate_tasks(-1)
            else:
                self._snapshots += [_SnapshotData(prev.element, prev.tasks)]
            self.swap_lifetime_proxy_array()

    def receive_and_update(self, element : ElementData):
        """
        Create or update the live frame with the incoming data.
        """
        global _receive_times
        _receive_times[element.seqno] = time.perf_counter()
        # check if this is the first time receiving data
        if not self.has_data():
            self._snapshots += [_SnapshotData(element, None)]
        else:
            snap = self._snapshots[-1] # live frame is the last
            snap.element = element
            snap.tasks.invalidate()
        
        self.validate_tasks(-1)
        self.swap_lifetime_proxy_array()

    def set_settings(self, settings : Settings):
        """
        The settings have changed, we must invalidate all old tasks and populate proxy array with fresh tasks.
        """
        self.settings = settings
        if self.has_data():
            for i in range(len(self._snapshots)):
                tasks = self._snapshots[i].tasks
                if tasks is not None:
                    tasks.invalidate()
                self.validate_tasks(i)
            self.swap_lifetime_proxy_array()
    
    @ensure_main_thread
    def compute_done_callback(self, done):
        self.compute_done.emit()
        for i in range(len(self._snapshots)):
            self.validate_tasks(i)
        self.swap_lifetime_proxy_array()

    def swap_lifetime_proxy_array(self):
        self.lifetime_image.data = create_lifetime_image_proxy(self)

    def get_tasks_list(self):
        return [snapshot.tasks for snapshot in self._snapshots]

    def get_task(self, step):
        return self.get_tasks_list()[step] if -len(self._snapshots) <= step < len(self._snapshots) else None

    def get_tau_axis_size(self):
        return self.shape[-1]

    def get_image_shape(self):
        """Returns shape of the image (height, width)"""
        return self.shape[-3:-1]

    def get_num_phasors(self):
        return np.prod(self.get_image_shape())

    def has_data(self):
        return bool(self._snapshots)

    def live_index(self):
        return len(self._snapshots) - 1
    
    def get_frame_no(self, step):
        elem = self.get_element(step)
        return elem.seqno if elem is not None else None

    def get_photon_count(self, step) -> np.ndarray:
        """
        Returns the effective photon count at the given step
        that should be used for computation. If no snapshot exists
        at the step, returns a NaN filled array instead.
        """
        elem = self.get_element(step)
        if elem is not None:
            if self.settings.delta_snapshots and step != 0:
                prev_elem = self.get_element(step - 1)
                if prev_elem is not None:
                    return elem.frame - prev_elem.frame
            return elem.frame
        return np.broadcast_to(np.array([np.nan]), self.shape)

    def get_element(self, step):
        if -len(self._snapshots) <= step < len(self._snapshots):
            return self._snapshots[step].element
        return None
