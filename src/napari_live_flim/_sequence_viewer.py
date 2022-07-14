import time
from concurrent.futures import Future
from dataclasses import dataclass
from functools import wraps
from time import time
from typing import TYPE_CHECKING, List

import flimlib
import logging
import numpy as np
from napari.layers import Image

from scipy.spatial import KDTree
from superqt import ensure_main_thread

from ._widget import executor

if TYPE_CHECKING:
    from ._series_viewer import SeriesViewer

from ._constants import *
from ._dataclasses import *
from .gather_futures import gather_futures


# adapted from stackoverflow.com :)
def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        logging.debug(f"Function {f.__name__} took {te-ts:2.4f} seconds")
        return result
    return wrap

class ComputeTask:
    def __init__(self, step : int, sequence_viewer : "SequenceViewer"):
        self._valid = True
        self._step = step
        self._sequence_viewer = sequence_viewer # need this to retrieve the most recent

        self.intensity = None
        self.lifetime_image = None
        self.phasor = None
        self.phasor_quadtree = None
        self.phasor_image = None
        self.phasor_face_color = None
        self.done = None

    # TODO is there a way to ensure start and cancel don't happen at the same time?
    # or is this even necessary based on where it gets called

    def start(self):
        if self._sequence_viewer is not None and not self.all_started(): # dont start more than once
            photon_count = self._sequence_viewer.get_photon_count(self._step)
            params = self._sequence_viewer.series_viewer.params
            filters = self._sequence_viewer.series_viewer.filters

            self.intensity = executor.submit(compute_intensity, photon_count)
            self.lifetime_image = executor.submit(compute_lifetime_image, photon_count, self.intensity, params, filters)
            self.phasor = executor.submit(compute_phasor, photon_count, params)
            self.phasor_quadtree = executor.submit(compute_phasor_quadtree, self.phasor)
            self.phasor_image = executor.submit(compute_phasor_image, self.phasor)
            self.phasor_face_color = executor.submit(compute_phasor_face_color, self.intensity)
            self.done = gather_futures(self.intensity, self.lifetime_image, self.phasor, self.phasor_quadtree, self.phasor_image, self.phasor_face_color)
            self.done.add_done_callback(self._sequence_viewer.compute_done_callback)

    def cancel(self):
        if self.all_started(): # if looking at an old snapshot
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
            raise ValueError("old proxy must have the same shape as this")
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

def create_lifetime_image_proxy(sequence_viewer : "SequenceViewer", dtype=np.float32):
    tasks_list = sequence_viewer.get_tasks_list()

    image_shape = sequence_viewer.get_image_shape() + (3,) # rgb
    possible_old_proxy = sequence_viewer.lifetime_image.data
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
def compute_lifetime_image(photon_count : np.ndarray, intensity_future : Future[np.ndarray], params : FlimParams, filters : DisplayFilters):
    period = params.period
    fstart = params.fit_start if params.fit_start < photon_count.shape[-1] else photon_count.shape[-1]
    fend =  params.fit_end if params.fit_end <= photon_count.shape[-1] else photon_count.shape[-1]
    rld = flimlib.GCI_triple_integral_fitting_engine(period, photon_count, fit_start=fstart, fit_end=fend, compute_fitted=False, compute_residuals=False)
    tau = rld.tau
    
    intensity = intensity_future.result()
    tau[intensity < filters.min_intensity] = 0
    intensity = normalize(intensity)
    tau[rld.chisq > filters.max_chisq] = 0
    # negative lifetimes are not valid
    tau[tau<0] = 0
    tau[tau > filters.max_tau] = 0
    tau = normalize(tau, bound=COLOR_DEPTH)
    np.nan_to_num(tau, copy=False)
    tau = tau.astype(int)
    tau[tau >= COLOR_DEPTH] = COLOR_DEPTH - 1 # this value is used to index into the colormap
    intensity_scaled_tau = COLORMAP[tau]
    intensity_scaled_tau[...,0] *= intensity
    intensity_scaled_tau[...,1] *= intensity
    intensity_scaled_tau[...,2] *= intensity
    return intensity_scaled_tau

def compute_intensity(photon_count : np.ndarray) -> np.ndarray:
    return np.nansum(photon_count, axis=-1)

def compute_phasor_image(phasor_future : Future[np.ndarray]):
    phasor = phasor_future.result()
    return phasor.reshape(-1,phasor.shape[-1])

def compute_phasor_face_color(intensity_future : Future[np.ndarray]):
    intensity = normalize(intensity_future.result())
    phasor_intensity = intensity.ravel() * PHASOR_OPACITY_FACTOR
    color = np.broadcast_to(1.0, phasor_intensity.shape)
    return np.asarray([color,color,color,phasor_intensity]).T

def normalize(data : np.ndarray, bound=1) -> np.ndarray:
    maximum = np.nanmax(data)
    return data * (bound / maximum) if maximum > 0 else np.zeros_like(data)

@timing
def compute_phasor(photon_count : np.ndarray, params : FlimParams):
    period = params.period
    fstart = params.fit_start if params.fit_start < photon_count.shape[-1] else photon_count.shape[-1]
    fend =  params.fit_end if params.fit_end <= photon_count.shape[-1] else photon_count.shape[-1]
    # about 0.5 sec for 256x256x256 data
    phasor = flimlib.GCI_Phasor(period, photon_count, fit_start=fstart, fit_end=fend, compute_fitted=False, compute_residuals=False, compute_chisq=False)
    #reshape to work well with mapping / creating the phasor plot. Result has shape (height, width, 2)
    return np.round(np.dstack([(1 - phasor.v) * PHASOR_SCALE, phasor.u * PHASOR_SCALE])).astype(int)

def compute_phasor_quadtree(phasor_future : Future[np.ndarray]):
    phasor = phasor_future.result()
    return KDTree(phasor.reshape(-1, phasor.shape[-1]))

@dataclass
class SnapshotData:
    photon_count : np.ndarray
    tasks : ComputeTask

class SequenceViewer:
    def __init__(self, image : Image, shape, series_viewer : "SeriesViewer"):
        self.snapshots : List[SnapshotData]= []
        self.series_viewer = series_viewer
        self.shape = shape
        self.lifetime_image = image

    def validate_tasks(self, index):
        tasks = self.snapshots[index].tasks
        if tasks is None:
            self.snapshots[index].tasks = ComputeTask(index, self)
        elif not tasks.is_valid() and not tasks.is_running():
            self.snapshots[index].tasks = ComputeTask(index, self)
            tasks.cancel()

    def snap(self):
        if self.has_data():
            prev = self.snapshots[-1]
            self.snapshots += [SnapshotData(prev.photon_count, prev.tasks)]
            self.swap_lifetime_proxy_array()

    def receive_and_update(self, photon_count : np.ndarray):
        # for now, we ignore all but the first channel
        photon_count = photon_count[tuple([0] * (photon_count.ndim - 3))]
        # check if this is the first time receiving data
        if not self.has_data():
            self.snapshots += [SnapshotData(photon_count, None)]
        else:
            snap = self.snapshots[-1] # live frame is the last
            snap.photon_count = photon_count
            snap.tasks.invalidate()
        
        self.validate_tasks(-1)
        self.swap_lifetime_proxy_array()

    def update(self):
        if self.has_data():
            for i in range(len(self.snapshots)):
                tasks = self.snapshots[i].tasks
                if tasks is not None:
                    tasks.invalidate()
                self.validate_tasks(i)
            self.swap_lifetime_proxy_array()
    
    @ensure_main_thread
    def compute_done_callback(self, done):
        self.series_viewer.update_displays()
        for i in range(len(self.snapshots)):
            self.validate_tasks(i)
        self.swap_lifetime_proxy_array()

    def swap_lifetime_proxy_array(self):
        self.lifetime_image.data = create_lifetime_image_proxy(self)

    def get_tasks_list(self):
        return [snapshot.tasks for snapshot in self.snapshots]

    def get_task(self, step):
        return self.get_tasks_list()[step] if -len(self.snapshots) <= step < len(self.snapshots) else None

    def get_tau_axis_size(self):
        return self.shape[-1]

    def get_image_shape(self):
        """returns shape of the image (height, width)"""
        return self.shape[-3:-1]

    def get_num_phasors(self):
        return np.prod(self.get_image_shape())

    def has_data(self):
        return bool(self.snapshots)

    def live_index(self):
        return len(self.snapshots) - 1

    def get_photon_count(self, step) -> np.ndarray:
        if -len(self.snapshots) <= step < len(self.snapshots):
            if self.series_viewer.params.delta_snapshots and step != 0 and step != -len(self.snapshots):
                return self.snapshots[step].photon_count - self.snapshots[step - 1].photon_count
            else:
                return self.snapshots[step].photon_count
        return np.broadcast_to(np.array([np.nan]), self.shape)
