from __future__ import annotations

import inspect
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import asdict
from typing import List

import flimlib
import numpy as np
from napari import Viewer
from napari.components.viewer_model import ViewerModel
from napari.layers import Image, Layer, Points, Shapes
from napari.utils.events import Event
from napari.qt import QtViewer
from superqt import ensure_main_thread
from vispy.scene.visuals import Text

from ._constants import *
from ._dataclasses import *
from ._series_viewer import ComputeTask, SeriesViewer
from ._widget import *
from .gather_futures import gather_futures
from .plot_widget import Fig
from .timing import timing

from ._flim_receiver import FlimReceiver

class Controller():
    """
    The "controller" class for the Napari Live Flim plugin. Holds on to references
    of the UI elements each which are allowed to communicate back via Signals.
    """
    def __init__(self, napari_viewer : Viewer):

        self._flim_params = None
        self._display_settings = None
        self._delta_snapshots = None
        self._settings_filepath = None
        self._port = None

        self.flim_receiver = FlimReceiver()
        self.exposed_lifetime_image = None
        self.live_series_viewer = None
        
        self.lifetime_viewer = napari_viewer
        self.qt_main_window = napari_viewer.window._qt_window
        self.qt_lifetime_viewer = napari_viewer.window.qt_viewer
        self.phasor_viewer = ViewerModel(title="Phasor Viewer")
        self.qt_phasor_viewer = QtViewer(self.phasor_viewer)
        self.qt_phasor_viewer.window().setMinimumHeight(200)
        self.phasor_dock_widget = self.lifetime_viewer.window.add_dock_widget(self.qt_phasor_viewer, area="bottom")

        #add the phasor circle
        phasor_circle = [[0, 0.5],[0.5, 0.5]]
        black_box = [[0, -0.5],[0, 1.5], [-1, 1.5], [-1, -0.5]]
        x_axis_line = [[0,0], [0,1]]
        phasor_shapes_layer = self.phasor_viewer.add_shapes(phasor_circle, shape_type="ellipse", face_color="", scale=[-PHASOR_SCALE, PHASOR_SCALE], opacity=1.0, edge_width=5/PHASOR_SCALE)
        phasor_shapes_layer.add_rectangles(black_box, edge_width=0, face_color="black")
        phasor_shapes_layer.add_lines(x_axis_line)
        phasor_shapes_layer.editable = False
        # empty phasor data
        self.phasor_image = self.phasor_viewer.add_points(None, name="Phasor", edge_width=0, size=3/PHASOR_SCALE, scale=[-PHASOR_SCALE, PHASOR_SCALE])
        zoom_viewer(self.qt_phasor_viewer, ((0,-PHASOR_SCALE/2), (PHASOR_SCALE, PHASOR_SCALE/2)))
        self.phasor_image.editable = False

        ph_ctrls = self.qt_phasor_viewer.dockLayerControls
        ph_ctrls.name = "Phasor Layer Controls"
        ph_ctrls.setWindowTitle("Phasor Layer Controls")
        lt_ctrls = inspect.unwrap(self.qt_lifetime_viewer.dockLayerControls)
        self.qt_main_window.tabifyDockWidget(lt_ctrls, ph_ctrls)
        ph_list = self.qt_phasor_viewer.dockLayerList
        ph_list.name = "Phasor Layer List"
        ph_list.setWindowTitle("Phasor Layer List")
        lt_list = inspect.unwrap(self.qt_lifetime_viewer.dockLayerList)
        self.qt_main_window.tabifyDockWidget(lt_list, ph_list)
        
        self.qt_lifetime_viewer.canvas.events.mouse_press.connect(self.switch_to_lifetime_controls)
        self.qt_phasor_viewer.canvas.events.mouse_press.connect(self.switch_to_phasor_controls)
        self.qt_phasor_viewer.canvas.events.mouse_move.connect(self.display_phasor_mouse_pos)
        self.lifetime_viewer.layers.events.connect(self.validate_exposed_lifetime_image)
        self.lifetime_viewer.layers.events.removed.connect(lambda e: self.cleanup_removed_selection(e.value, self.lifetime_viewer, self.phasor_viewer))
        self.phasor_viewer.layers.events.removed.connect(lambda e: self.cleanup_removed_selection(e.value, self.phasor_viewer, self.lifetime_viewer))
        self.lifetime_viewer.dims.events.current_step.connect(self.update_displays)
        self.lifetime_viewer.dims.events.order.connect(self.update_displays)
        self.lifetime_viewer.grid.events.enabled.connect(self.update_displays)

        self.port_widget = PortSelection()
        self.port_widget.port_line_edit.textChanged.connect(lambda p : setattr(self, "port", p))

        self.flim_params_widget = FlimParamsWidget()
        self.flim_params_widget.changed.connect(lambda fp : setattr(self, "flim_params", fp))

        self.display_settings_widget = DisplaySettingsWidget()
        self.display_settings_widget.changed.connect(lambda ds : setattr(self, "display_settings", ds))

        self.actions_widget = ActionsWidget()
        self.actions_widget.snap_button.clicked.connect(lambda: self.snap())
        self.actions_widget.delta_snapshots.toggled.connect(lambda ds : setattr(self, "delta_snapshots", ds))
        self.actions_widget.hide_plots_button.clicked.connect(lambda: self.hide_plots())
        self.actions_widget.show_plots_button.clicked.connect(lambda: self.show_plots())
        self.actions_widget.new_lifetime_selection_button.clicked.connect(lambda: self.create_lifetime_select_layer())
        self.actions_widget.new_phasor_selection_button.clicked.connect(lambda: self.create_phasor_select_layer())

        self.save_settings_widget = SaveSettingsWidget()
        self.save_settings_widget.filepath.textChanged.connect(lambda text : setattr(self, "settings_filepath", text))
        self.save_settings_widget.save.clicked.connect(self.save_settings)
        self.save_settings_widget.save_as.clicked.connect(self.save_as_settings)
        self.save_settings_widget.open.clicked.connect(self.open_settings)

        self.flim_receiver.new_series.connect(self.new_series)
        self.flim_receiver.new_element.connect(self.new_element)
        self.flim_receiver.end_series.connect(self.end_series)

        self.flim_params = DEFAULT_FLIM_PARAMS
        self.display_settings = DEFAULT_DISPLAY_SETTINGS
        self.delta_snapshots = DEFAULT_DELTA_SNAPSHOTS
        self.settings_filepath = DEFAULT_SETTINGS_FILEPATH
        self.port = DEFAULT_PORT
        
        self.reset_current_step()

        # color generator used for color coordinated selections
        def color_gen():
            while True:
                for i in COLOR_DICT.keys():
                    yield COLOR_DICT[i]
        self.colors = color_gen()
        
        self.create_lifetime_select_layer()

    @property
    def port(self):
        return self._port
    
    @port.setter
    def port(self, value : str | int):
        if self.port_widget.port_line_edit.text() != str(value):
            self.port_widget.port_line_edit.setText(str(value))
        if self._port != int(value):
            if str.isnumeric(str(value)) and int(value) > 1023 and int(value) < 65536:
                self._port = int(value)
                self.port_widget.set_valid()
                self.flim_receiver.start_receiving(int(value))
            else:
                self._port = None
                self.port_widget.set_invalid()
                self.flim_receiver.stop_receiving()

    @property
    def flim_params(self):
        return self._flim_params

    @flim_params.setter
    def flim_params(self, value : FlimParams):
        if self._flim_params != value:
            self._flim_params = value
            self.update_settings()
        if self.flim_params_widget.values() != value:
            self.flim_params_widget.setValues(value)

    @property
    def display_settings(self):
        return self._display_settings

    @display_settings.setter
    def display_settings(self, value : DisplaySettings):
        if self._display_settings != value:
            self._display_settings = value
            self.update_settings()
        if self.display_settings_widget.values() != value:
            self.display_settings_widget.setValues(value)

    @property
    def delta_snapshots(self):
        return self._delta_snapshots

    @delta_snapshots.setter
    def delta_snapshots(self, value : bool):
        if self._delta_snapshots != value:
            self._delta_snapshots = value
            self.update_settings()
        if self.actions_widget.delta_snapshots.isChecked() != value:
            self.actions_widget.delta_snapshots.setChecked(value)

    @property
    def settings_filepath(self):
        return self._settings_filepath

    @settings_filepath.setter
    def settings_filepath(self, value : str):
        self._settings_filepath = value
        if self.save_settings_widget.filepath.text() != value:
            self.save_settings_widget.filepath.setText(value)

    def switch_to_phasor_controls(self, event=None):
        self.show_layer_controls()
        self.qt_phasor_viewer.dockLayerControls.raise_()
        self.qt_phasor_viewer.dockLayerList.raise_()

    def switch_to_lifetime_controls(self, event=None):
        self.show_layer_controls()
        self.qt_lifetime_viewer.dockLayerControls.raise_()
        self.qt_lifetime_viewer.dockLayerList.raise_()

    def display_phasor_mouse_pos(self, event : Event):
        self.phasor_viewer.text_overlay.visible = True
        pos = self.phasor_viewer.cursor.position # (y, x)
        g = pos[1] / PHASOR_SCALE
        s = - pos[0] / PHASOR_SCALE
        self.phasor_viewer.text_overlay.text = f"g = {g}\ns = {s}"
    
    def tear_down(self):
        """
        Stop the `FlimReceiver` and `__del__` references to Qt objects.
        This step is neccessary for tests that create instances of this class
        """
        self.flim_receiver.stop_receiving()
        try:
            del self.qt_lifetime_viewer
            del self.qt_phasor_viewer
            del self.qt_main_window
        except NameError:
            logging.error("Failed to tear down controller or already was")

    @ensure_main_thread
    def new_series(self, series_metadata : SeriesMetadata):
        self.port_widget.disable_editing()
        if self.flim_params == DEFAULT_FLIM_PARAMS:
            tau_axis_size = series_metadata.shape[-1]
            self.flim_params = FlimParams(
                    period=self.flim_params.period,
                    fit_start=tau_axis_size // 3,
                    fit_end=(tau_axis_size * 2 ) // 3,
                )
        self.setup_series_viewer(series_metadata)

    @ensure_main_thread
    def new_element(self, element_data : ElementData):
        self.receive_and_update(element_data)

    @ensure_main_thread
    def end_series(self):
        self.port_widget.enable_editing()

    def save_settings(self):
        path = Path(self.settings_filepath).absolute()
        logging.info(f"saving parameters to {path}")
        with open(path, "w") as outfile:
            opts_dict = asdict(self.get_settings())
            opts_dict["version"] = SETTINGS_VERSION
            json.dump(opts_dict, outfile, indent=4)

    def save_as_settings(self):
        fs = FileSelector()
        self.settings_filepath = fs.save_file(self.settings_filepath)
        self.save_settings()

    def open_settings(self):
        fs = FileSelector()
        self.settings_filepath = fs.open_file(self.settings_filepath)
        path = Path(self.settings_filepath).absolute()
        logging.info(f"loading parameters from {path}")
        with open(path, "r") as infile:
            opts_dict = json.load(infile)
            assert opts_dict.pop("version", None) == SETTINGS_VERSION
            stgs = Settings.from_dict(opts_dict)
            self.set_settings(stgs)

    def get_lifetime_layers(self, include_invisible=True) -> List[Layer]:
        """
        Returns a list of layers that contain a `SeriesViewer` object in the metadata.
        """
        layer_list = self.lifetime_viewer.layers
        return [layer for layer in layer_list if (include_invisible or layer.visible) and has_series_viewer(layer)]

    def get_series_viewers(self, include_invisible=True) -> List[SeriesViewer]:
        """
        Returns a list of active `SeriesViewers` objects.
        """
        lifetime_layers = self.get_lifetime_layers(include_invisible=include_invisible)
        return [get_series_viewer(layer) for layer in lifetime_layers]

    def validate_exposed_lifetime_image(self, event : Event):
        typ = event.type
        if typ == "reordered" or typ == "visible" or typ == "removed":
            layers = self.get_lifetime_layers(include_invisible=False)
            if len(layers) == 1:
                if self.get_exposed_series_viewer() is not get_series_viewer(layers[-1]):
                    self.exposed_lifetime_image = layers[-1]
                    self.update_displays()
            else:
                if self.exposed_lifetime_image is not None:
                    self.exposed_lifetime_image = None
                    self.update_displays()

    def cleanup_removed_selection(self, layer : Layer, viewer : Viewer, co_viewer : Viewer):
        """
        Called when a selection or co-selection is being removed. If `layer` is a selection, 
        removes the co-selection, if `layer` is a co-selection, removes the selection that
        contains it. In both cases, the decay plot is also removed.

        Parameters
        ----------
        layer : Layer
            The layer being removed
        viewer : Viewer
            The viewer that the `layer` belongs to
        co_viewer : Viewer
            The viewer that the `layer` does not belong to
        """
        if has_selection(layer):
            # a selection is being remoevd
            selection = get_selection(layer)
            try:
                co_viewer.layers.remove(selection.co_selection)
                viewer.window.remove_dock_widget(selection.decay_plot.dock_widget)
            except ValueError:
                pass # already removed
        else:
            # A co-selection is being removed
            selection = get_selection_from_co_selection(layer, co_viewer)
            if selection is not None:
                try:
                    co_viewer.layers.remove(selection.selection)
                    co_viewer.window.remove_dock_widget(selection.decay_plot.dock_widget)
                except ValueError:
                    pass # already removed

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
        """
        Update phasor plot and selections with results if available
        """
        series_viewer = self.get_exposed_series_viewer()
        frame_no = None
        if self.should_show_displays() and series_viewer is not None:
            frame_no = series_viewer.get_frame_no(self.get_current_step())
            step = self.get_current_step()
            tasks = series_viewer.get_task(step)
            if tasks is not None and tasks.all_done():
                set_points(self.phasor_image, tasks.phasor_image.result(timeout=0))
                try:
                    self.phasor_image.face_color = tasks.phasor_face_color.result(timeout=0)
                except RuntimeError as e:
                    logging.error(f"Attempting to set phasor face color resulted in {e}")
                self.phasor_image.selected_data = {}
        else:
            set_points(self.phasor_image, None)

        if frame_no is not None:
            self.lifetime_viewer.text_overlay.visible = True
            self.lifetime_viewer.text_overlay.text = f"frame {frame_no}"
        else:
            self.lifetime_viewer.text_overlay.visible = False

        self.update_selections()

    def get_exposed_series_viewer(self) -> SeriesViewer | None:
        return get_series_viewer(self.exposed_lifetime_image) if self.exposed_lifetime_image is not None else None

    def setup_series_viewer(self, series_metadata : SeriesMetadata):
        """
        sets up the series viewer after receiving the new_series signal
        At this point we know what shape the incoming data will be
        """
        shape = series_metadata.shape
        image_shape = shape[-3:-1]
        
        lifetime_layers = self.get_lifetime_layers()
        for layer in lifetime_layers:
                layer.visible = False
        self.lifetime_viewer.dims.set_current_step(0, 0)

        name = str(series_metadata.series_no) + "-" + str(series_metadata.port)
        sel = self.lifetime_viewer.layers.selection.copy()
        image = self.lifetime_viewer.add_image(EMPTY_RGB_IMAGE, rgb=True, name=name)
        self.lifetime_viewer.layers.selection = sel
        # Move new lifetime image to the end of the current lifetime layers
        self.lifetime_viewer.layers.move(len(self.lifetime_viewer.layers) - 1, len(lifetime_layers))
        zoom_viewer(self.qt_lifetime_viewer, ((0,0), image_shape))
        self.live_series_viewer = SeriesViewer(image, shape, self.get_settings())
        self.live_series_viewer.compute_done.connect(self.update_displays)
        set_series_viewer(image, self.live_series_viewer)
        
        self.exposed_lifetime_image = image

    def receive_and_update(self, element : ElementData):
        if self.live_series_viewer is not None:
            self.live_series_viewer.receive_and_update(element)
        else:
            logging.error("Received data before processing start series message")

    def snap(self):
        """
        Take a snapshot of the current live frame if any. If currently viewing the
        live (largest index) frame, the viewer's current step will be incremented
        continue viewing the live frame. Else, will stay on the current step.
        """
        sv = self.live_series_viewer
        if sv is not None:
            scroll_next = sv.live_index() == self.get_current_step()
            sv.snap()
            if scroll_next:
                self.lifetime_viewer.dims.set_current_step(0, sv.live_index())

    def update_settings(self):
        stgs = self.get_settings()
        for series_viewer in self.get_series_viewers():
            series_viewer.set_settings(stgs)

    def show_layer_controls(self):
        self.qt_lifetime_viewer.dockLayerControls.show()
        self.qt_lifetime_viewer.dockLayerList.show()
        self.qt_phasor_viewer.dockLayerControls.show()
        self.qt_phasor_viewer.dockLayerList.show()

    def get_settings(self):
        return Settings(
            delta_snapshots=self.delta_snapshots,
            flim_params=self.flim_params,
            display_settings=self.display_settings,
        )

    def set_settings(self, settings : Settings):
        # Calls setter methods of the following properties. May cause a few redunadnt updates
        self.delta_snapshots = settings.delta_snapshots
        self.flim_params = settings.flim_params
        self.display_settings = settings.display_settings

    def show_plots(self):
        for layer in self.lifetime_viewer.layers:
            show_decay_plot(layer)
        for layer in self.phasor_viewer.layers:
            show_decay_plot(layer)

        try:
            if self.phasor_dock_widget.isHidden():
                self.phasor_dock_widget.show()
        except RuntimeError:
            logging.info("Phasor viewer dock widget was deleted. Attempting to restore...")
            self.lifetime_viewer.window.add_dock_widget(self.qt_phasor_viewer, area="bottom")

        self.show_layer_controls()

    def hide_plots(self):
        for layer in self.lifetime_viewer.layers:
            hide_decay_plot(layer)
        for layer in self.phasor_viewer.layers:
            hide_decay_plot(layer)
        try:
            self.phasor_dock_widget.hide()
        except RuntimeError:
            logging.info("Phasor viewer dock widget was deleted. Unable to hide it.")

    @ensure_main_thread
    def update_selections_callback(self, done):
        self.update_selections()

    def update_selections(self):
        for selection in get_selections(self.lifetime_viewer):
            selection.update()
        for selection in get_selections(self.phasor_viewer):
            selection.update()
    
    def create_lifetime_select_layer(self):
        viewer = self.lifetime_viewer
        co_viewer = self.phasor_viewer
        color = next(self.colors)
        select_layer = viewer.add_shapes(DEFUALT_LIFETIME_SELECTION, shape_type="ellipse", name="Selection", face_color=color+"7f", edge_width=0)
        sel = co_viewer.layers.selection.copy()
        co_selection = co_viewer.add_points(None, name="Correlation", size=1/PHASOR_SCALE, face_color=color, edge_width=0, scale=[-PHASOR_SCALE, PHASOR_SCALE])
        co_viewer.layers.selection = sel
        co_selection.editable = False
        decay_plot = CurveFittingPlot(self.lifetime_viewer, scatter_color=color)
        set_selection(select_layer, LifetimeSelectionMetadata(select_layer, co_selection, decay_plot, self))
        select_layer.mouse_drag_callbacks.append(select_shape_drag)
        select_layer.events.data.connect(update_selection_callback)
        select_layer.events.visible.connect(update_selection_callback)
        select_layer.mode = "select"
        return select_layer

    # TODO most of this code is duplicate of above method
    def create_phasor_select_layer(self):
        viewer = self.phasor_viewer
        co_viewer = self.lifetime_viewer
        color = next(self.colors)
        select_layer = viewer.add_shapes(DEFUALT_PHASOR_SELECTION, shape_type="ellipse", name="Selection", face_color=color+"7f", edge_width=0, scale=[-1, 1])
        sel = co_viewer.layers.selection.copy()
        co_selection = co_viewer.add_points(None, name="Correlation", size=1, face_color=color, edge_width=0)
        co_viewer.layers.selection = sel
        co_selection.editable = False
        decay_plot = CurveFittingPlot(self.lifetime_viewer, scatter_color=color)
        set_selection(select_layer, PhasorSelectionMetadata(select_layer, co_selection, decay_plot, self))
        select_layer.mouse_drag_callbacks.append(select_shape_drag)
        select_layer.events.data.connect(update_selection_callback)
        select_layer.events.visible.connect(update_selection_callback)
        select_layer.mode = "select"
        return select_layer

def compute_fits(photon_count, params : "FlimParams"):
    period = params.period
    fstart = params.fit_start if params.fit_start < photon_count.shape[-1] else photon_count.shape[-1]
    fend =  params.fit_end if params.fit_end <= photon_count.shape[-1] else photon_count.shape[-1]
    rld = flimlib.GCI_triple_integral_fitting_engine(period, photon_count, fit_start=fstart, fit_end=fend, compute_residuals=False)
    param_in = [rld.Z, rld.A, rld.tau]
    lm = flimlib.GCI_marquardt_fitting_engine(period, photon_count, param_in, fit_start=fstart, fit_end=fend, compute_residuals=False, compute_covar=False, compute_alpha=False, compute_erraxes=False)
    return rld, lm

def zoom_viewer(viewer : QtViewer, bounds):
    state = {"rect": bounds}
    viewer.view.camera.set_state(state)

class CurveFittingPlot():
    """
    A class responsible for plotting the decay plots within a dock widget
    """
    #TODO add transform into log scale
    def __init__(self, viewer : Viewer, scatter_color="magenta"):
        self.fig = Fig()
        # add a docked figure
        self.dock_widget = viewer.window.add_dock_widget(self.fig, area="bottom")
        # TODO remove access to private member
        self.dock_widget._close_btn = False
        
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
        """
        Updates the decay plot with the finished `SelectionComputeTask`.
        Raises a `TimeoutError` if the task provided is not finished yet
        """
        selection = selection_task.selection.result(timeout=0)
        params = selection_task.params
        
        rld_selected = selection.rld
        lm_selected = selection.lm
        period = params.period
        fit_start = params.fit_start
        fit_end = params.fit_end

        time = np.linspace(0, lm_selected.fitted.size * params.period, lm_selected.fitted.size, endpoint=False, dtype=np.float32)
        fit_time = time[fit_start:fit_end]
        # account for a (vispy?) bug where setting as size zero data does not properly clear the drawn curves
        if len(fit_time) > 0: 
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
        
        # attempt to autoscale based on data (ignore start/end lines)
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

        controller = selection_metadata.controller
        series_viewer = controller.get_exposed_series_viewer()
        if controller.should_show_displays() and series_viewer is not None:
            stp = controller.get_current_step()
            photon_count = series_viewer.get_photon_count(stp)
            tasks = series_viewer.get_task(stp)
        else:
            photon_count = EMPTY_PHOTON_COUNT
            tasks = None

        self.params = controller.flim_params
        if len(selection_metadata.selection.data) > 0 and selection_metadata.selection.visible:
            mask_result = selection_metadata.compute_mask(photon_count)
        else:
            mask_result = None
        self.selection = EXECUTOR.submit(selection_metadata.compute_selection, mask_result, tasks, photon_count, self.params)
        self.done = gather_futures(self.selection)
        self.done.add_done_callback(selection_metadata.update_callback)            

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
    """
    An abstract class that exposes common fields for selection computation and result display.
    Subclasses are to be inserted into Napari Shapes Layer metadata in order to designate that layer
    as a selection layer.
    """
    def __init__(self, selection : Shapes, co_selection : Points, decay_plot : CurveFittingPlot, controller : Controller):
        self.selection = selection
        self.co_selection = co_selection
        self.decay_plot = decay_plot
        self.controller = controller
        self.tasks = None
    
    @abstractmethod
    def compute_selection(self, mask_result: MaskResult, tasks: ComputeTask, photon_count: np.ndarray, params : "FlimParams") -> SelectionResult:
        pass

    @abstractmethod
    def compute_mask(self, photon_count) -> MaskResult | None:
        pass

    @ensure_main_thread
    def update_callback(self, done):
        """
        Update the selection results (on the main thread) 
        after the compute thread finishes"""
        self._update()

    def _update(self):
        """
        Restarts tasks if invalid and not currently running. Otherwise, if the
        results are available, display them.
        """
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
        """
        Invalidates the old selection results and restarts them. 
        Called externally: example - when the selection is dragged.
        """
        if self.tasks is not None:
            self.tasks.invalidate()
        self._update()

class LifetimeSelectionMetadata(SelectionMetadata):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def compute_mask(self, photon_count):
        masks = self.selection.to_masks(photon_count.shape[-3:-1]).astype(bool)
        union_mask = np.logical_or.reduce(masks)
        return MaskResult(mask=union_mask, extrema=None)

    @timing(name="compute_selection (lifetime image)")
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
        extrema = np.ceil(self.selection._extent_data).astype(int) # the private field since `extent` is a `cached_property`
        bounding_shape = extrema[1] - extrema[0] + 1 # add one since extremas are inclusive
        offset=extrema[0]
        # use of private field `_data_view` since the shapes.py `to_masks()` fails to recognize offset
        masks = self.selection._data_view.to_masks(mask_shape=bounding_shape, offset=offset)
        union_mask = np.logical_or.reduce(masks)
        return MaskResult(extrema, union_mask)

    @timing(name="compute_selection (phasor image)")
    def compute_selection(self, mask_result: MaskResult, tasks: ComputeTask, photon_count: np.ndarray, params : "FlimParams") -> SelectionResult:
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
                phasor = (tasks.phasor.result(timeout=0) * PHASOR_SCALE).astype(int)
                for point in bounded_points:
                    bounded_phasor = phasor[tuple(point)]
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
                    histogram = np.mean(photon_count[points_indexer], axis=0)
                    rld, lm = compute_fits(histogram, params)
                    return SelectionResult(histogram=histogram, points=points, rld=rld, lm=lm)

        histogram = np.broadcast_to(np.array([np.nan]), (photon_count.shape[-1],))
        co_selection = None
        rld, lm = compute_fits(histogram, params) # TODO these are just gonna fail and return Nan results
        return SelectionResult(histogram=histogram, points=co_selection, rld=rld, lm=lm)

def set_points(points_layer : Points, points : np.ndarray):
    try:
        points_layer.data = points if points is None or len(points) else None
    except OverflowError as e:
        # there seems to be a bug in napari with an overflow error
        logging.error(f"Attempting to set data of a points layer resulted in {e}")
    points_layer.selected_data = {}

def set_series_viewer(layer : Image, series_viewer : SeriesViewer):
    layer.metadata[KEY_SERIES_VIEWER] = series_viewer

def get_series_viewer(layer : Image) -> SeriesViewer | None:
    return layer.metadata[KEY_SERIES_VIEWER] if has_series_viewer(layer) else None

def has_series_viewer(layer : Image):
    return KEY_SERIES_VIEWER in layer.metadata

def set_selection(layer : Shapes, selection_metadata : SelectionMetadata):
    layer.metadata[KEY_SELECTION] = selection_metadata

def get_selection(layer : Shapes) -> SelectionMetadata | None:
    return layer.metadata[KEY_SELECTION] if has_selection(layer) else None

def has_selection(layer : Shapes):
    return KEY_SELECTION in layer.metadata

def get_selections(viewer : Viewer):
    return [get_selection(layer) for layer in viewer.layers if has_selection(layer)]

def get_selection_from_co_selection(co_selection : Layer, viewer : Viewer):
        """
        Returns the `SelectionMetadata` object contained in one of the layers of the `viewer`
        whose co-selection is the given `co_selection`. If unable to find, returns `None`
        """
        for test_sel in get_selections(viewer):
            if test_sel.co_selection is co_selection:
                return test_sel
        return None

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

def update_selection_callback(event : Event):
    event_layer = event.sources[0]
    update_selection(event_layer)

def show_decay_plot(layer : Shapes):
    selection = get_selection(layer)
    if selection is not None:
        selection.decay_plot.dock_widget.show()

def hide_decay_plot(layer : Shapes):
    selection = get_selection(layer)
    if selection is not None:
        selection.decay_plot.dock_widget.hide()


