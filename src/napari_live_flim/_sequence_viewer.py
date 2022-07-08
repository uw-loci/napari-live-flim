import time
from concurrent.futures import Future
from dataclasses import dataclass
from functools import wraps
from time import time
from typing import TYPE_CHECKING, List

import flimlib
import numpy as np

from scipy.spatial import KDTree
from superqt import ensure_main_thread

from ._widget import executor

if TYPE_CHECKING:
    from ._series_viewer import SeriesViewer

from ._constants import *
from ._dataclasses import *
from .gather_futures import gather_futures


# copied from stackoverflow.com :)
def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print(f"Function {f.__name__} took {te-ts:2.4f} seconds")
        return result
    return wrap

class ComputeTask:
    def __init__(self, step, sequence_viewer : "SequenceViewer"):
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

    @ensure_main_thread # start and cancel must not happen on different threads
    def start(self):
        if not self.all_started(): # dont start more than once
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

    @ensure_main_thread # start and cancel must not happen on different threads
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

    def __init__(self, sequence_viewer : "SequenceViewer", dtype=np.float32):
        tasks_list = sequence_viewer.get_tasks_list()
        if not len(tasks_list):
            raise ValueError # At least for now, dont allow empty

        self._arrays = tasks_list
        self._dtype = dtype
        self._image_shape = sequence_viewer.get_image_shape() + (3,) # rgb
        old_proxy = sequence_viewer.lifetime_image.data
        if not isinstance(old_proxy, LifetimeImageProxy):
            self.most_recent = np.zeros(self._image_shape, dtype=dtype)
        else:
            self.most_recent = old_proxy.most_recent

    def set_tasks_list(self, tasks_list : List[ComputeTask]):
        self._arrays = tasks_list

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
            shape = (dim0,) + self._image_shape
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
            print("Displaying done data!!!!!!!!!!")
            self.most_recent = task.lifetime_image.result(timeout=0)
        else:
            print("#############starting tasks####################")
            task.start()
        return self.most_recent[slices]

    def __array__(self):
        print("################## NAPARI IS REQUESTING THE FULL ARRAY #####################")
        return self[:]
    
    def __len__(self):
        return np.prod(self.shape)

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
    maxi = np.nanmax(intensity)
    intensity = intensity * (1 / maxi) if maxi > 0 else np.zeros_like(intensity)
    tau[rld.chisq > filters.max_chisq] = 0
    # negative lifetimes are not valid
    tau[tau<0] = 0
    tau[tau > filters.max_tau] = 0
    maxt = np.nanmax(tau)
    tau = tau * (COLOR_DEPTH / maxt) if maxt > 0 else np.zeros_like(tau)
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

def compute_phasor_image(phasor : Future[np.ndarray]):
    return phasor.result().reshape(-1,phasor.result().shape[-1])

def compute_phasor_face_color(intensity : Future[np.ndarray]):
    it = intensity.result()
    it = it / it.max()
    phasor_intensity = it.ravel() * PHASOR_OPACITY_FACTOR
    color = np.broadcast_to(1.0, phasor_intensity.shape)
    return np.asarray([color,color,color,phasor_intensity]).T

@timing
def compute_phasor(photon_count : np.ndarray, params : FlimParams):
    period = params.period
    fstart = params.fit_start if params.fit_start < photon_count.shape[-1] else photon_count.shape[-1]
    fend =  params.fit_end if params.fit_end <= photon_count.shape[-1] else photon_count.shape[-1]
    # about 0.5 sec for 256x256x256 data
    phasor = flimlib.GCI_Phasor(period, photon_count, fit_start=fstart, fit_end=fend, compute_fitted=False, compute_residuals=False, compute_chisq=False)
    #reshape to work well with mapping / creating the image
    #TODO can i have the last dimension be tuple? this would simplify indexing later
    return np.round(np.dstack([(1 - phasor.v) * PHASOR_SCALE, phasor.u * PHASOR_SCALE])).astype(int)

def compute_phasor_quadtree(phasor : Future[np.ndarray]):
    return KDTree(phasor.result().reshape(-1, phasor.result().shape[-1]))

@dataclass
class SnapshotData:
    photon_count : np.ndarray
    tasks : ComputeTask

class SequenceViewer:
    def __init__(self, image, shape, series_viewer : "SeriesViewer"):
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
            index = len(self.snapshots)
            self.snapshots += [SnapshotData(prev.photon_count, ComputeTask(index, self))]

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
        self.lifetime_image.data = LifetimeImageProxy(self)

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

    def get_photon_count(self, step) -> np.ndarray:
        if -len(self.snapshots) <= step < len(self.snapshots):
            if self.series_viewer.params.delta_snapshots and step != 0:
                return self.snapshots[step].photon_count - self.snapshots[step - 1].photon_count
            else:
                return self.snapshots[step].photon_count
        return np.broadcast_to(np.array([np.nan]), self.shape)
