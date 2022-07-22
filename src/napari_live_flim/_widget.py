import json
import sys
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
import matplotlib

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.colorbar import Colorbar
from matplotlib.figure import Figure
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

from superqt import ensure_main_thread
from napari import Viewer
from PyQt5 import QtGui
from qtpy.QtCore import QObject, Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QWidget,
    QComboBox,
)

executor = ThreadPoolExecutor()

from ._constants import *
from ._dataclasses import *
from ._flim_receiver import FlimReceiver
from ._series_viewer import SeriesViewer

flim_receiver = FlimReceiver()

class FlimViewer(QWidget):
    # QWidget.__init__ can optionally request the napari viewer instance
    def __init__(self, napari_viewer : Viewer):
        super().__init__()

        self.lifetime_viewer = napari_viewer
        self.phasor_viewer = Viewer(title="Phasor Viewer")
        self.phasor_viewer.window.qt_viewer.dockLayerList.setVisible(False)
        self.phasor_viewer.window.qt_viewer.dockLayerControls.setVisible(False)
        def close_phasor_viewer():
            try:
                self.phasor_viewer.close()
            except RuntimeError:
                logging.warn(f"Failed to close phasor viewer or already closed!")
        self.destroyed.connect(close_phasor_viewer)

        self.layout = QFormLayout()
        self.setLayout(self.layout)

        self.port_widget = PortSelection()
        self.port_widget.port_selected.connect(flim_receiver.start_receiving)
        self.port_widget.port_removed.connect(flim_receiver.stop_receiving)
        self.layout.addRow(self.port_widget.group)

        self.flim_params_widget = FlimParamsWidget()
        self.flim_params_widget.changed.connect(lambda p: self.series_viewer.set_params(p))
        self.layout.addRow(self.flim_params_widget.group)

        self.display_filters_widget = DisplayFiltersWidget()
        self.display_filters_widget.changed.connect(lambda f: self.series_viewer.set_filters(f))
        self.layout.addRow(self.display_filters_widget.group)

        self.actions_widget = ActionsWidget()
        self.actions_widget.snap_button.clicked.connect(lambda: self.series_viewer.snap())
        self.actions_widget.delta_snapshots.toggled.connect(lambda d: self.series_viewer.set_delta_snapshots(d))
        self.actions_widget.hide_plots_button.clicked.connect(lambda: self.series_viewer.hide_plots())
        self.actions_widget.show_plots_button.clicked.connect(lambda: self.series_viewer.show_plots())
        self.actions_widget.new_lifetime_selection_button.clicked.connect(lambda: self.series_viewer.create_lifetime_select_layer())
        self.actions_widget.new_phasor_selection_button.clicked.connect(lambda: self.series_viewer.create_phasor_select_layer())
        self.layout.addRow(self.actions_widget.group)

        self.save_settings_widget = SaveSettingsWidget()
        self.layout.addRow(self.save_settings_widget.group)

        self.series_viewer = SeriesViewer(
            self.lifetime_viewer,
            self.phasor_viewer,
            self.actions_widget.delta_snapshots.isChecked(),
            self.flim_params_widget.values(),
            self.display_filters_widget.values(),
        )

        flim_receiver.new_series.connect(self.new_series)
        flim_receiver.new_element.connect(self.new_element)
        flim_receiver.end_series.connect(self.end_series)
        self.destroyed.connect(lambda : flim_receiver.stop_receiving())
        
    
    @ensure_main_thread
    def new_series(self, series_metadata : SeriesMetadata):
        self.port_widget.disable_editing()
        self.options_widget.flim_params_widget.initialize_fit_range(series_metadata)
        self.series_viewer.setup_sequence_viewer(series_metadata)
    
    @ensure_main_thread
    def new_element(self, element_data : ElementData):
        self.series_viewer.receive_and_update(element_data)

    @ensure_main_thread
    def end_series(self):
        self.port_widget.enable_editing()

    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:
        super().closeEvent(a0)
        print(type(a0))

    def hideEvent(self, a0: QtGui.QHideEvent) -> None:
        super().hideEvent(a0)
        print(type(a0))

class PortSelection(QObject):
    
    port_selected = Signal(int)
    port_removed = Signal()

    def __init__(self) -> None:
        """
        Port selection widget
        """
        super(PortSelection, self).__init__()

        self.group = QGroupBox()
        self.label = QLabel("Port")

        self.port_line_edit = QLineEdit("")
        self.port_line_edit.textChanged.connect(self._setPortEnabled)

        self.valid_label = QLabel()
        self._set_invalid()

        # create widget layout
        self.layout = QHBoxLayout()
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.port_line_edit)
        self.layout.addWidget(self.valid_label)
        self.group.setLayout(self.layout)

    def _setPortEnabled(self, a0: str):
        """Private method serving as an enable/disable mechanism for the Add button widget.
        This way, only valid ports will be input
        """
        if str.isnumeric(a0) and int(a0) > 1023 and int(a0) < 65536:
            self._set_valid()
            self.port_selected.emit(int(self.port_line_edit.text()))
        else:
            self._set_invalid()
            self.port_removed.emit()

    def _set_valid(self):
        self.valid_label.setText("✓")
        self.valid_label.setStyleSheet("QLabel { color : green }")

    def _set_invalid(self):
        self.valid_label.setText("X")
        self.valid_label.setStyleSheet("QLabel { color : red }")

    def disable_editing(self):
        self.port_line_edit.setReadOnly(True)
        self.port_line_edit.setStyleSheet("QLineEdit { color: grey }")
    
    def enable_editing(self):
        self.port_line_edit.setReadOnly(False)
        self.port_line_edit.setStyleSheet("QLineEdit { color: white }")

class FileSelector(QWidget):
    def save_file(self, filepath):
        filepath = QFileDialog.getSaveFileName(self, "Save - Options", filepath, "Json (*.json)")
        return filepath[0]
    def open_file(self, filepath):
        filepath = QFileDialog.getOpenFileName(self, "Open - Options", filepath, "Json (*.json)")
        return filepath[0]

class SaveSettingsWidget(QObject):
    changed_options = Signal()

    def __init__(self) -> None:
        super(SaveSettingsWidget, self).__init__()

        self.layout = QFormLayout()
        self.group = QGroupBox("Options")
        self.group.setLayout(self.layout)

        self.filepath = QLineEdit("./options.json")
        self.save = QPushButton("Save")
        self.save.clicked.connect(lambda: self.save_options())
        self.layout.addRow(self.filepath, self.save)

        self.open = QPushButton("Open")
        self.open.setMinimumWidth(110)
        self.open.clicked.connect(lambda: self.open_options())
        self.save_as = QPushButton("Save as")
        self.save_as.clicked.connect(lambda: self.save_as_options())
        self.layout.addRow(self.open, self.save_as)

    def save_options(self):
        path = Path(self.filepath.text()).absolute()
        logging.info(f"saving parameters to {path}")
        with open(path, "w") as outfile:
            opts_dict = {
                "version" : OPTIONS_VERSION,
                "delta_snapshots" : self.delta_snapshots.isChecked(),
                "flim_params" : asdict(self.flim_params_widget.values()),
                "display_filters" : asdict(self.display_filters_widget.values())
            }
            json.dump(opts_dict, outfile, indent=4)

    def save_as_options(self):
        fs = FileSelector()
        self.filepath.setText(fs.save_file(self.filepath.text()))
        self.save_options()

    def open_options(self):
        fs = FileSelector()
        path = Path(fs.open_file(self.filepath.text())).absolute()
        self.filepath.setText(str(path))
        logging.info(f"loading parameters from {path}")
        with open(path, "r") as infile:
            opts_dict = json.load(infile)
            assert opts_dict["version"] == OPTIONS_VERSION
            self.delta_snapshots.setChecked(opts_dict["delta_snapshots"])
            self.flim_params_widget.setValues(FlimParams(**opts_dict["flim_params"]))
            self.display_filters_widget.setValues(DisplayFilters(**opts_dict["display_filters"]))

class FlimParamsWidget(QObject):

    changed = Signal(FlimParams)

    def __init__(self) -> None:
        super(FlimParamsWidget, self).__init__()

        self.group = QGroupBox("FLIM Parameters")
        self.layout = QFormLayout()
        self.group.setLayout(self.layout)

        self.period = QDoubleSpinBox()
        self.period.setMinimum(sys.float_info.min)
        self.period.setMaximum(MAX_VALUE)
        self.period.setSingleStep(0.01)
        self.period.valueChanged.connect(self.changed_callback)
        self.layout.addRow("Period (ns)", self.period)

        self.fit_start = QSpinBox()
        self.fit_start_label = QLabel("Fit Start")
        self.fit_start.setValue(0)
        self.fit_start.setRange(0, 1)
        self.fit_start.valueChanged.connect(self.changed_callback)
        fsl = lambda: self.fit_start_label.setText("Fit Start= {:.2f}ns".format(self.fit_start.value() * self.period.value()))
        self.fit_start.valueChanged.connect(fsl)
        self.period.valueChanged.connect(fsl)
        self.layout.addRow(self.fit_start_label, self.fit_start)

        self.fit_end = QSpinBox()
        self.fit_end_label = QLabel("Fit End")
        self.fit_end.setValue(1)
        self.fit_end.setRange(1, MAX_VALUE)
        self.fit_end.valueChanged.connect(self.changed_callback)
        fel = lambda: self.fit_end_label.setText("Fit End= {:.2f}ns".format(self.fit_end.value() * self.period.value()))
        self.fit_end.valueChanged.connect(fel)
        self.period.valueChanged.connect(fel)
        self.layout.addRow(self.fit_end_label, self.fit_end)

        self.fit_start.valueChanged.connect(lambda a0 : self.fit_end.setMinimum(a0 + 1))
        self.fit_end.valueChanged.connect(lambda a0 : self.fit_start.setMaximum(a0 - 1))

        self.period.setValue(DEFAULT_PERIOD)

        self.is_changed = False # if false, estimate fit range when data arrives
        #self.set_flim_params(FlimParams(.4, 23, 57, True))
    
    def changed_callback(self):
            self.is_changed = True
            self.changed.emit(self.values())

    def initialize_fit_range(self, series_metadata : SeriesMetadata):
        if not self.is_changed:
            tau_axis_size = series_metadata.shape[-1]
            self.fit_end.setValue((tau_axis_size * 2 ) // 3)
            self.fit_start.setValue(tau_axis_size // 3)

    # not used
    def estimate_fit_range(self, photon_count : np.ndarray):
        summed_photon_count = photon_count.reshape(-1, photon_count.shape[-1]).sum(axis=0)
        
        fstart = int(np.argmax(summed_photon_count)) #estimate fit start as the max in the data
        fend = int(np.max(np.nonzero(summed_photon_count)) + 1) # estimate fit end as bounding the last nonzero data
        self.fit_start.setValue(fstart)
        self.fit_end.setValue(fend)

    def values(self):
        return FlimParams(
            self.period.value(),
            self.fit_start.value(),
            self.fit_end.value(),
        )

    def setValues(self, flim_params : FlimParams):
        self.period.setValue(flim_params.period)
        self.fit_start.setValue(flim_params.fit_start)
        self.fit_end.setValue(flim_params.fit_end)
        
class DisplayFiltersWidget(QObject):

    changed = Signal(DisplayFilters)

    def __init__(self) -> None:
        super(DisplayFiltersWidget, self).__init__()

        self.group = QGroupBox("Display Filters")
        self.layout = QFormLayout()
        self.group.setLayout(self.layout)

        def changed_callback():
            self.changed.emit(self.values())
            self.update_colorbar()

        self.max_chisq = QDoubleSpinBox()
        self.max_chisq.setRange(0, MAX_VALUE)
        self.max_chisq.setValue(DEFUALT_MAX_CHISQ)
        self.max_chisq.valueChanged.connect(changed_callback)
        self.layout.addRow("Max χ2", self.max_chisq)

        self.min_tau = QDoubleSpinBox()
        self.min_tau.setRange(0, DEFAULT_MAX_TAU)
        self.min_tau.setValue(0)
        self.min_tau.setSingleStep(0.1)
        self.min_tau.setDecimals(3)
        self.min_tau.valueChanged.connect(changed_callback)
        self.layout.addRow("Min Lifetime (ns)", self.min_tau)

        self.max_tau = QDoubleSpinBox()
        self.max_tau.setRange(0, MAX_VALUE)
        self.max_tau.setValue(DEFAULT_MAX_TAU)
        self.max_tau.setSingleStep(0.1)
        self.max_tau.setDecimals(3)
        self.max_tau.valueChanged.connect(changed_callback)
        self.layout.addRow("Max Lifetime (ns)", self.max_tau)

        self.min_tau.valueChanged.connect(lambda a0 : self.max_tau.setMinimum(a0))
        self.max_tau.valueChanged.connect(lambda a0 : self.min_tau.setMaximum(a0))

        self.colormap = QComboBox()
        self.colormap.addItems(COLORMAPS.keys())
        if "BH_compat" in COLORMAPS.keys():
            self.colormap.setCurrentText("BH_compat")
        self.colormap.currentTextChanged.connect(changed_callback)
        self.layout.addRow("Colormap", self.colormap)

        with plt.style.context("dark_background"):
            self.colorbar_widget = FigureCanvasQTAgg(Figure(constrained_layout=True))
            fig = self.colorbar_widget.figure
            ax = fig.add_subplot()
            fig.subplots_adjust(bottom=0.5)

            self.mappable = ScalarMappable(cmap=COLORMAPS[self.colormap.currentText()])
            self.mappable.set_clim(self.min_tau.value(), self.max_tau.value())
            colorbar = fig.colorbar(self.mappable, cax=ax, orientation="horizontal")
            colorbar.minorticks_on()

            self.colorbar_widget.setFixedWidth(230)
            self.colorbar_widget.setFixedHeight(50)
        self.layout.addRow(self.colorbar_widget)

    def update_colorbar(self):
        self.mappable.set_cmap(COLORMAPS[self.colormap.currentText()])
        self.mappable.set_clim(self.min_tau.value(), self.max_tau.value())
        self.colorbar_widget.draw_idle()

    def values(self):
        return DisplayFilters(
            self.max_chisq.value(),
            self.min_tau.value(),
            self.max_tau.value(),
            self.colormap.currentText(),
        )

    def setValues(self, display_filters : DisplayFilters):
        self.max_chisq.setValue(display_filters.max_chisq)
        self.min_tau.setValue(display_filters.min_tau)
        self.max_tau.setValue(display_filters.max_tau)
        self.colormap.setCurrentText(display_filters.colormap)

class ActionsWidget(QObject):
    def __init__(self) -> None:
        super(ActionsWidget, self).__init__()

        self.layout = QFormLayout()
        self.group = QGroupBox()
        self.group.setLayout(self.layout)

        self.snap_button = QPushButton("Snapshot")
        self.snap_button.setMinimumWidth(110)
        self.delta_snapshots = QCheckBox("Delta Snapshots")
        self.layout.addRow(self.snap_button, self.delta_snapshots)

        self.hide_plots_button = QPushButton("Hide Plots")
        self.hide_plots_button.setMinimumWidth(110)
        self.show_plots_button = QPushButton("Show Plots")
        self.layout.addRow(self.hide_plots_button, self.show_plots_button)

        self.new_lifetime_selection_button = QPushButton("New Lifetime\nSelection")
        self.new_lifetime_selection_button.setMinimumWidth(110)
        self.new_phasor_selection_button = QPushButton("New Phasor\nSelection")
        self.layout.addRow(self.new_lifetime_selection_button, self.new_phasor_selection_button)
        