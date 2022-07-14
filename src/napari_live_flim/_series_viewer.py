from __future__ import annotations
import copy
from typing import List

import flimlib
from napari import Viewer
from napari.utils.events import Event
from napari.layers import Layer, Image, Shapes, Points
import numpy as np
from .plot_widget import Fig
from ._widget import executor

from superqt import ensure_main_thread
from vispy.scene.visuals import Text

from abc import ABC, abstractmethod

from ._constants import *
from .gather_futures import gather_futures

from ._dataclasses import *
from ._sequence_viewer import SequenceViewer, ComputeTask

class SeriesViewer():
    def __init__(self, lifetime_viewer : Viewer, phasor_viewer : Viewer, params : FlimParams, filters : DisplayFilters):
        self.params = params
        self.filters = filters

        self.lifetime_viewer = lifetime_viewer
        self.exposed_lifetime_image = None
        self.live_sequence_viewer = None
        self.lifetime_viewer.layers.events.connect(self.validate_exposed_lifetime_image)
        self.lifetime_viewer.dims.events.current_step.connect(self.update_displays)
        self.lifetime_viewer.dims.events.order.connect(self.update_displays)
        self.lifetime_viewer.grid.events.enabled.connect(self.update_displays)
        
        self.reset_current_step()

        self.phasor_viewer = phasor_viewer
        autoscale_viewer(self.phasor_viewer, (PHASOR_SCALE, PHASOR_SCALE))
        #add the phasor circle
        phasor_circle = np.asarray([[PHASOR_SCALE, 0.5 * PHASOR_SCALE],[0.5 * PHASOR_SCALE,0.5 * PHASOR_SCALE]])
        x_axis_line = np.asarray([[PHASOR_SCALE,0],[PHASOR_SCALE,PHASOR_SCALE]])
        phasor_shapes_layer = self.phasor_viewer.add_shapes([phasor_circle, x_axis_line], shape_type=["ellipse","line"], face_color="",)
        phasor_shapes_layer.editable = False
        # empty phasor data
        self.phasor_image = self.phasor_viewer.add_points(None, name="Phasor", edge_width=0, size=3)
        self.phasor_image.editable = False
        
        # color generator used for color coordinated selections
        def color_gen():
            while True:
                for i in COLOR_DICT.keys():
                    yield COLOR_DICT[i]
        self.colors = color_gen()
        
        self.create_lifetime_select_layer()

    def get_lifetime_layers(self, include_invisible=True) -> List[Layer]:
        layer_list = self.lifetime_viewer.layers
        ret = []
        for layer in layer_list:
            if (include_invisible or layer.visible) and get_sequence_viewer(layer) is not None:
                ret += [layer]
        return ret

    def get_sequence_viewers(self, include_invisible=True) -> List[SequenceViewer]:
        return [get_sequence_viewer(layer) for layer in self.get_lifetime_layers(include_invisible=include_invisible)]

    def validate_exposed_lifetime_image(self, event : Event):
        if event.type == "reordered" or event.type == "visible":
            layers = self.get_lifetime_layers(include_invisible=False)
            if len(layers) == 1:
                if self.get_exposed_sequence_viewer() is not get_sequence_viewer(layers[-1]):
                    self.exposed_lifetime_image = layers[-1]
                    self.update_displays()
            else:
                if self.exposed_lifetime_image is not None:
                    self.exposed_lifetime_image = None
                    self.update_displays()

    def set_params(self, params : FlimParams):
        self.params = params
        self.update_all()

    def set_filters(self, filters : DisplayFilters):
        self.filters = filters
        self.update_all()

    def get_current_step(self):
        return self.lifetime_viewer.dims.current_step[0]

    def reset_current_step(self):
        self.lifetime_viewer.dims.set_current_step(0,0)
    
    def should_show_displays(self):
        if 0 in self.lifetime_viewer.dims.displayed:
            return False
        if self.lifetime_viewer.grid.enabled:
            return False
        return True

    def update_displays(self):
        sequence_viewer = self.get_exposed_sequence_viewer()
        if self.should_show_displays() and sequence_viewer is not None:
            step = self.get_current_step()
            tasks = sequence_viewer.get_task(step)
            if tasks is not None and tasks.all_done():
                set_points(self.phasor_image, tasks.phasor_image.result(timeout=0))
                try:
                    self.phasor_image.face_color = tasks.phasor_face_color.result(timeout=0)
                except RuntimeError:
                    print(tasks.phasor_face_color.result(timeout=0))
                self.phasor_image.selected_data = {}
        else:
            set_points(self.phasor_image, None)

        self.update_selections()

    def get_exposed_sequence_viewer(self) -> SequenceViewer | None:
        return get_sequence_viewer(self.exposed_lifetime_image) if self.exposed_lifetime_image is not None else None

    def setup_sequence_viewer(self, series_metadata : SeriesMetadata):
        """
        setup occurs after receiving the new_series signal
        At this point we know what shape the incoming data will be
        """
        shape = series_metadata.shape
        image_shape = shape[-3:-1]
        
        for layer in self.get_lifetime_layers():
                layer.visible = False

        name = str(series_metadata.series_no) + "-" + str(series_metadata.port)
        sel = self.lifetime_viewer.layers.selection.copy()
        image = self.lifetime_viewer.add_image(EMPTY_RGB_IMAGE, rgb=True, name=name)
        self.lifetime_viewer.layers.selection = sel
        self.lifetime_viewer.layers.move(len(self.lifetime_viewer.layers) - 1, len(self.get_lifetime_layers()))
        autoscale_viewer(self.lifetime_viewer, image_shape)
        self.live_sequence_viewer = SequenceViewer(image, shape, self)
        set_sequence_viewer(image, self.live_sequence_viewer)
        
        self.exposed_lifetime_image = image

    # called after new data arrives
    def receive_and_update(self, element : ElementData):
        self.live_sequence_viewer.receive_and_update(element.frame)

    def snap(self):
        sv = self.live_sequence_viewer
        if sv is not None:
            scroll_next = sv.live_index() == self.get_current_step()
            sv.snap()
            if scroll_next:
                self.lifetime_viewer.dims.set_current_step(0, sv.live_index())

    def update_all(self):
        for sequence_viewer in self.get_sequence_viewers():
            sequence_viewer.update()
        

    @ensure_main_thread
    def update_selections_callback(self, done):
        self.update_selections()

    # TODO selections should only select the current viewed channel (if channels are added)
    def update_selections(self):
        for layer in self.lifetime_viewer.layers:
            update_selection(layer)
        for layer in self.phasor_viewer.layers:
            update_selection(layer)
    
    def create_lifetime_select_layer(self):
        viewer = self.lifetime_viewer
        co_viewer = self.phasor_viewer
        color = next(self.colors)
        select_layer = viewer.add_shapes(DEFUALT_LIFETIME_SELECTION, shape_type="ellipse", name="Selection", face_color=color+"7f", edge_width=0)
        sel = co_viewer.layers.selection.copy()
        co_selection = co_viewer.add_points(None, name="Correlation", size=1, face_color=color, edge_width=0)
        co_viewer.layers.selection = sel
        co_selection.editable = False
        decay_plot = CurveFittingPlot(viewer, scatter_color=color)
        set_selection(select_layer, LifetimeSelectionMetadata(select_layer, co_selection, decay_plot, self))
        select_layer.mouse_drag_callbacks.append(select_shape_drag)
        select_layer.events.data.connect(handle_new_shape)
        select_layer.mode = "select"
        viewer.window.qt_viewer.dockLayerList.setVisible(True)
        viewer.window.qt_viewer.dockLayerControls.setVisible(True)
        return select_layer

    # TODO most of this code is duplicate of above method
    def create_phasor_select_layer(self):
        viewer = self.phasor_viewer
        co_viewer = self.lifetime_viewer
        color = next(self.colors)
        select_layer = viewer.add_shapes(DEFUALT_PHASOR_SELECTION, shape_type="ellipse", name="Selection", face_color=color+"7f", edge_width=0)
        sel = co_viewer.layers.selection.copy()
        co_selection = co_viewer.add_points(None, name="Correlation", size=1, face_color=color, edge_width=0)
        co_viewer.layers.selection = sel
        co_selection.editable = False
        decay_plot = CurveFittingPlot(viewer, scatter_color=color)
        set_selection(select_layer, PhasorSelectionMetadata(select_layer, co_selection, decay_plot, self))
        select_layer.mouse_drag_callbacks.append(select_shape_drag)
        select_layer.events.data.connect(handle_new_shape)
        select_layer.mode = "select"
        viewer.window.qt_viewer.dockLayerList.setVisible(True)
        viewer.window.qt_viewer.dockLayerControls.setVisible(True)
        return select_layer

def compute_fits(photon_count, params : "FlimParams"):
    period = params.period
    fstart = params.fit_start if params.fit_start < photon_count.shape[-1] else photon_count.shape[-1]
    fend =  params.fit_end if params.fit_end <= photon_count.shape[-1] else photon_count.shape[-1]
    rld = flimlib.GCI_triple_integral_fitting_engine(period, photon_count, fit_start=fstart, fit_end=fend, compute_residuals=False)
    param_in = [rld.Z, rld.A, rld.tau]
    lm = flimlib.GCI_marquardt_fitting_engine(period, photon_count, param_in, fit_start=fstart, fit_end=fend, compute_residuals=False, compute_covar=False, compute_alpha=False, compute_erraxes=False)
    return rld, lm

def autoscale_viewer(viewer : Viewer, shape):
    state = {"rect": ((0, 0), shape)}
    viewer.window.qt_viewer.view.camera.set_state(state)

class CurveFittingPlot():
    #TODO add transform into log scale
    def __init__(self, viewer : Viewer, scatter_color="magenta"):
        self.fig = Fig()
        # add a docked figure
        self.dock_widget = viewer.window.add_dock_widget(self.fig, area="bottom")
        # TODO remove access to private member
        self.dock_widget._close_btn = False
        # TODO float button crashes the entire app. Couldn"t find a way to remove it. Fix the crash?
        
        # get a handle to the plotWidget
        self.ax = self.fig[0, 0]
        self.lm_curve = self.ax.plot(None, color="g", marker_size=0, width=2)
        self.rld_curve = self.ax.plot(None, color="r", marker_size=0, width=2)
        self.scatter_color = scatter_color
        self.data_scatter = self.ax.scatter(None, size=1, edge_width=0, face_color=scatter_color)
        self.fit_start_line = self.ax.plot(None, color="b", marker_size=0, width=2)
        self.fit_end_line = self.ax.plot(None, color="b", marker_size=0, width=2)
        self.rld_info = Text(None, parent=self.ax.view, color="r", anchor_x="right", font_size = FONT_SIZE)
        self.lm_info = Text(None, parent=self.ax.view, color="g", anchor_x="right", font_size = FONT_SIZE)
    
    def update_with_selection(self, selection_task : "SelectionComputeTask"):
        selection = selection_task.selection.result(timeout=0)
        params = selection_task.params
        
        rld_selected = selection.rld
        lm_selected = selection.lm
        period = params.period
        fit_start = params.fit_start
        fit_end = params.fit_end

        time = np.linspace(0, lm_selected.fitted.size * params.period, lm_selected.fitted.size, endpoint=False, dtype=np.float32)
        fit_time = time[fit_start:fit_end]
        if len(fit_time) > 0: # a bug where setting as size zero data does not properly clear the drawn curves
            self.lm_curve.set_data((fit_time, lm_selected.fitted[fit_start:fit_end]))
            self.rld_curve.set_data((fit_time, rld_selected.fitted[fit_start:fit_end]))
        else:
            self.lm_curve.set_data(([0],[0]))
            self.rld_curve.set_data(([0],[0]))
        self.data_scatter.set_data(np.array((time, selection.histogram)).T, size=3, edge_width=0, face_color=self.scatter_color)
        self.rld_info.pos = self.ax.view.size[0], self.rld_info.font_size
        self.rld_info.text = "RLD | chisq = " + "{:.2e}".format(float(rld_selected.chisq)) + ", tau = " + "{:.2e}".format(float(rld_selected.tau))
        self.lm_info.pos = self.ax.view.size[0], self.rld_info.font_size*3
        self.lm_info.text = "LMA | chisq = " + "{:.2e}".format(float(lm_selected.chisq)) + ", tau = " + "{:.2e}".format(float(lm_selected.param[2]))
        
        # autoscale based on data (ignore start/end lines)
        self.fit_start_line.set_data(np.zeros((2,1)))
        self.fit_end_line.set_data(np.zeros((2,1)))
        try:
            self.ax.autoscale()
        except ValueError:
            pass
        self.fit_start_line.set_data(([fit_start * period, fit_start * period], self.ax.camera._ylim))
        self.fit_end_line.set_data(([fit_end * period, fit_end * period], self.ax.camera._ylim))



class SelectionComputeTask:
    def __init__(self, selection_metadata : "SelectionMetadata"):
        self._valid = True
        self.done = None

        series_viewer = selection_metadata.series_viewer
        sequence_viewer = series_viewer.get_exposed_sequence_viewer()
        if series_viewer.should_show_displays() and sequence_viewer is not None:
            stp = series_viewer.get_current_step()
            photon_count = sequence_viewer.get_photon_count(stp)
            tasks = sequence_viewer.get_task(stp)
        else:
            photon_count = EMPTY_PHOTON_COUNT
            tasks = None

        self.params = series_viewer.params

        mask_result = selection_metadata.compute_mask(photon_count)
        self.selection = executor.submit(selection_metadata.compute_selection, mask_result, tasks, photon_count, self.params)
        self.done = gather_futures(self.selection)
        self.done.add_done_callback(selection_metadata.update_callback)            

    @ensure_main_thread
    def cancel(self):
        if self.all_started(): # if looking at an old snapshot
            self.selection.cancel()
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

class SelectionMetadata(ABC):
    def __init__(self, selection : Shapes, co_selection : Points, decay_plot : CurveFittingPlot, series_viewer : SeriesViewer):
        self.selection = selection
        self.co_selection = co_selection
        self.decay_plot = decay_plot
        self.series_viewer = series_viewer
        self.tasks = None
    
    @abstractmethod
    def compute_selection(self, mask_result: MaskResult, tasks: ComputeTask, photon_count: np.ndarray, params : "FlimParams") -> SelectionResult:
        pass

    @abstractmethod
    def compute_mask(self, photon_count) -> MaskResult | None:
        pass

    @ensure_main_thread
    def update_callback(self, done):
        self._update()

    def _update(self):
        if self.tasks is None:
            self.tasks = SelectionComputeTask(self)
        if not self.tasks.is_valid() and not self.tasks.is_running():
            # restart selection tasks
            self.tasks.cancel()
            self.tasks = SelectionComputeTask(self)
        if self.tasks.all_done():
            #update selection displays
            self.decay_plot.update_with_selection(self.tasks)
            # TODO why does the following line take more than 50% of the runtime of this function?
            set_points(self.co_selection, self.tasks.selection.result(timeout=0).points)
            self.co_selection.editable = False
        
    def update(self):
        if self.tasks is not None:
            self.tasks.invalidate()
        self._update()

class LifetimeSelectionMetadata(SelectionMetadata):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def compute_mask(self, photon_count):
        if len(self.selection.data) == 0:
            return None
        masks = self.selection.to_masks(photon_count.shape[-3:-1]).astype(bool)
        union_mask = np.logical_or.reduce(masks)
        return MaskResult(mask=union_mask, extrema=None)

    def compute_selection(self, mask_result: MaskResult, tasks: ComputeTask, photon_count: np.ndarray, params : "FlimParams") -> SelectionResult:
        if mask_result is not None and tasks is not None and tasks.all_done():
            points = np.asarray(np.where(mask_result.mask)).T
            if(len(points) > 0):
                points_indexer = tuple(np.asarray(points).T)
                co_selection = tasks.phasor.result(timeout=0)[points_indexer]
                histogram = np.mean(photon_count[points_indexer],axis=0)
                rld, lm = compute_fits(histogram, params)
                return SelectionResult(histogram=histogram, points=co_selection, rld=rld, lm=lm)
        histogram = np.broadcast_to(np.array([np.nan]), (photon_count.shape[-1],))
        co_selection = None
        rld, lm = compute_fits(histogram, params) # TODO these are just gonna fail and return Nan results
        return SelectionResult(histogram=histogram, points=co_selection, rld=rld, lm=lm)

class PhasorSelectionMetadata(SelectionMetadata):
    def compute_mask(self, photon_count):
        if len(self.selection.data) == 0:
            return None
        extrema = np.ceil(self.selection._extent_data).astype(int) # the private field since `extent` is a `cached_property`
        bounding_shape = extrema[1] - extrema[0] + 1 # add one since extremas are inclusive
        offset=extrema[0]
        # use of private field `_data_view` since the shapes.py `to_masks()` fails to recognize offset
        masks = self.selection._data_view.to_masks(mask_shape=bounding_shape, offset=offset)
        union_mask = np.logical_or.reduce(masks)
        return MaskResult(extrema, union_mask)

    def compute_selection(self, mask_result: MaskResult, tasks: ComputeTask, photon_count: np.ndarray, params : "FlimParams") -> SelectionResult:
        phcpy = copy.copy(photon_count)
        if mask_result is not None and tasks is not None and tasks.all_done():
            extrema = mask_result.extrema
            mask = mask_result.mask
            bounding_center = np.mean(extrema, axis=0)
            bounding_shape = extrema[1] - extrema[0] + 1 # add one since extremas are inclusive
            bounding_radius = np.max(bounding_center - extrema[0]) # distance in the p = inf norm
            height, width = photon_count.shape[-3:-1]
            maxpoints = width * height
            distances, indices = tasks.phasor_quadtree.result(timeout=0).query(bounding_center, maxpoints, p=np.inf, distance_upper_bound=bounding_radius)
            n_indices = np.searchsorted(distances, np.inf)
            if n_indices > 0:
                indices = indices[0:n_indices]
                bounded_points = np.asarray([indices // height, indices % width]).T

                points = []
                offset=extrema[0]
                # use of private field `_data_view` since the shapes.py `to_masks()` fails to recognize offset
                for point in bounded_points:
                    bounded_phasor = tasks.phasor.result(timeout=0)[tuple(point)]
                    mask_indexer = tuple(bounded_phasor - offset)
                    # kd tree found a square bounding box. some of these points might be outside of the rectangular mask
                    if mask_indexer[0] < 0 or mask_indexer[1] < 0 or mask_indexer[0] >= bounding_shape[0] or mask_indexer[1] >= bounding_shape[1]:
                        continue
                    if mask[mask_indexer]:
                        points += [point]
                if points:
                    points = np.asarray(points)
                    if np.any(points < 0):
                        raise ValueError("Negative index encountered while indexing image layer. This is outside the image!")
                    points_indexer = tuple(points.T)
                    histogram = np.mean(phcpy[points_indexer], axis=0)
                    rld, lm = compute_fits(histogram, params)
                    return SelectionResult(histogram=histogram, points=points, rld=rld, lm=lm)

        histogram = np.broadcast_to(np.array([np.nan]), (photon_count.shape[-1],))
        co_selection = None
        rld, lm = compute_fits(histogram, params) # TODO these are just gonna fail and return Nan results
        return SelectionResult(histogram=histogram, points=co_selection, rld=rld, lm=lm)

def set_points(points_layer : Points, points : np.ndarray):
    try:
        points_layer.data = points if points is None or len(points) else None
    except OverflowError:
        # there seems to be a bug in napari with an overflow error
        pass
    points_layer.selected_data = {}

def set_sequence_viewer(layer : Image, sequence_viewer : SequenceViewer):
    layer.metadata["sequence_viewer"] = sequence_viewer

def get_sequence_viewer(layer : Image) -> SequenceViewer | None:
    return layer.metadata["sequence_viewer"] if "sequence_viewer" in layer.metadata else None

def set_selection(layer : Shapes, selection_metadata : SelectionMetadata):
    layer.metadata["selection"] = selection_metadata

def get_selection(layer : Shapes) -> SelectionMetadata | None:
    return layer.metadata["selection"] if "selection" in layer.metadata else None

def update_selection(layer : Shapes):
    selection = get_selection(layer)
    if selection is not None:   
        selection.update()

def select_shape_drag(layer : Shapes, event : Event):
    update_selection(layer)
    yield
    while event.type == "mouse_move":
        update_selection(layer)
        yield

def handle_new_shape(event : Event):
    event_layer = event._sources[0]
    # make sure to check if each of these operations has already been done since
    # changing the data triggers this event which may cause infinite recursion
    """
    # delete all shapes except for the new shape
    if len(event_layer.data) > 1 and event_layer.editable:
        event_layer.selected_data = range(0, len(event_layer.data) - 1)
        event_layer.remove_selected()
        event_layer.seleted_data = [0]
    """
    if len(event_layer.data) > 0:
        update_selection(event_layer)



