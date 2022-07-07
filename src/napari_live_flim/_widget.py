import json
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

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
        self.phasor_viewer = self.phasor_viewer = Viewer(title="Phasor Viewer")
        self.phasor_viewer.window.qt_viewer.dockLayerList.setVisible(False)
        self.phasor_viewer.window.qt_viewer.dockLayerControls.setVisible(False)

        #self.destroyed.connect(self.phasor_viewer.close)
        
        self.series_viewer = SeriesViewer(
            self.lifetime_viewer, 
            self.phasor_viewer, 
        )

        self.layout = QFormLayout()
        self.setLayout(self.layout)

        self.port_widget = PortSelection()
        self.port_widget.port_selected.connect(flim_receiver.start_receiving)
        self.port_widget.port_removed.connect(flim_receiver.stop_receiving)
        self.layout.addRow(self.port_widget.group)

        self.options_widget = OptionsWidget()
        self.options_widget.flim_params_widget.changed.connect(lambda p : self.series_viewer.set_params(p))
        self.options_widget.display_filters_widget.changed.connect(lambda f : self.series_viewer.set_filters(f))
        self.series_viewer.set_params(self.options_widget.flim_params_widget.values())
        self.series_viewer.set_filters(self.options_widget.display_filters_widget.values())
        self.layout.addRow(self.options_widget.group)

        self.selection_widget = SelectionWidget()
        self.selection_widget.new_lifetime_selection_button.clicked.connect(self.series_viewer.create_lifetime_select_layer)
        self.selection_widget.new_phasor_selection_button.clicked.connect(self.series_viewer.create_phasor_select_layer)
        self.layout.addRow(self.selection_widget.group)

        self.snap_widget = SnapWidget()
        self.snap_widget.snap_button.clicked.connect(self.series_viewer.snap)
        self.layout.addRow(self.snap_widget.group)

        flim_receiver.new_series.connect(lambda series_receiver : print(series_receiver.shape))
        flim_receiver.new_series.connect(lambda : self.port_widget.disable_editing())
        flim_receiver.new_series.connect(self.series_viewer.setup_sequence_viewer)
        flim_receiver.new_series.connect(self.options_widget.flim_params_widget.initialize_fit_range)
        flim_receiver.end_series.connect(lambda : print("end"))
        flim_receiver.end_series.connect(lambda : self.port_widget.enable_editing())
        self.destroyed.connect(lambda : flim_receiver.stop_receiving())
        
        flim_receiver.new_element.connect(self.series_viewer.receive_and_update)
        
        @self.lifetime_viewer.dims.events.current_step.connect
        def lifetime_slider_changed(event):
            #self.update_displays()
            #self.update_selections()
            pass
    
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

class OptionsWidget(QObject):
    changed_options = Signal()

    def __init__(self) -> None:
        super(OptionsWidget, self).__init__()

        self.layout = QFormLayout()
        self.group = QGroupBox("Options")
        self.group.setLayout(self.layout)

        self.flim_params_widget = FlimParamsWidget()
        self.layout.addRow(self.flim_params_widget.group)
        self.flim_params_widget.changed.connect(lambda : self.changed_options.emit())

        self.display_filters_widget = DisplayFiltersWidget()
        self.layout.addRow(self.display_filters_widget.group)
        self.display_filters_widget.changed.connect(lambda : self.changed_options.emit())

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
        print("saving parameters to", Path(self.filepath.text()).absolute)
        with open(self.filepath.text(),"w") as outfile:
            opts_dict = {
                "version" : OPTIONS_VERSION,
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
        self.filepath.setText(fs.open_file(self.filepath.text()))
        print("loading parameters from", Path(self.filepath.text()).absolute)
        with open(self.filepath.text(), "r") as infile:
            opts_dict = json.load(infile)
            assert opts_dict["version"] == OPTIONS_VERSION
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

        self.delta_snapshots = QCheckBox("Delta Snapshots")
        self.delta_snapshots.toggled.connect(self.changed_callback)
        self.layout.addRow(self.delta_snapshots)

        self.period.setValue(DEFAULT_PERIOD)

        self.is_changed = False # if false, estimate fit range when data arrives
        #self.set_flim_params(FlimParams(.4, 23, 57, True))
    
    def changed_callback(self):
            self.is_changed = True
            self.changed.emit(self.values())

    def initialize_fit_range(self, series_metadata : SeriesMetadata):
        print("initializing fit range")
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
            self.delta_snapshots.isChecked(),
        )

    def setValues(self, flim_params : FlimParams):
        self.period.setValue(flim_params.period)
        self.fit_start.setValue(flim_params.fit_start)
        self.fit_end.setValue(flim_params.fit_end)
        self.delta_snapshots.setChecked(flim_params.delta_snapshots)
        
class DisplayFiltersWidget(QObject):

    changed = Signal(DisplayFilters)

    def __init__(self) -> None:
        super(DisplayFiltersWidget, self).__init__()

        self.group = QGroupBox("Display Filters")
        self.layout = QFormLayout()
        self.group.setLayout(self.layout)

        changed_callback = lambda: self.changed.emit(self.values())

        self.min_intensity = QSpinBox()
        self.min_intensity.setRange(0, MAX_VALUE)
        self.min_intensity.setValue(DEFUALT_MIN_INTENSITY)
        self.min_intensity.valueChanged.connect(changed_callback)
        self.layout.addRow("Min Intensity", self.min_intensity)

        self.max_chisq = QDoubleSpinBox()
        self.max_chisq.setRange(0, MAX_VALUE)
        self.max_chisq.setValue(DEFUALT_MAX_CHISQ)
        self.max_chisq.valueChanged.connect(changed_callback)
        self.layout.addRow("Max χ2", self.max_chisq)

        self.max_tau = QDoubleSpinBox()
        self.max_tau.setRange(0, MAX_VALUE)
        self.max_tau.setValue(DEFAULT_MAX_TAU)
        self.max_tau.setSingleStep(0.1)
        self.max_tau.valueChanged.connect(changed_callback)
        self.layout.addRow("Max Lifetime", self.max_tau)

    def values(self):
        return DisplayFilters(
            self.min_intensity.value(),
            self.max_chisq.value(),
            self.max_tau.value(),
        )

    def setValues(self, display_filters : DisplayFilters):
        self.min_intensity.setValue(display_filters.min_intensity)
        self.max_chisq.setValue(display_filters.max_chisq)
        self.max_tau.setValue(display_filters.max_tau)

class SelectionWidget(QObject):
    def __init__(self) -> None:
        super(SelectionWidget, self).__init__()

        self.layout = QHBoxLayout()
        self.group = QGroupBox()
        self.group.setLayout(self.layout)

        self.new_lifetime_selection_button = QPushButton("New Lifetime\nSelection")
        self.layout.addWidget(self.new_lifetime_selection_button)

        self.new_phasor_selection_button = QPushButton("New Phasor\nSelection")
        self.layout.addWidget(self.new_phasor_selection_button)

class SnapWidget(QObject):
    def __init__(self) -> None:
        super(SnapWidget, self).__init__()

        self.layout = QHBoxLayout()
        self.group = QGroupBox()
        self.group.setLayout(self.layout)

        self.snap_button = QPushButton("Snap")
        self.layout.addWidget(self.snap_button)
        