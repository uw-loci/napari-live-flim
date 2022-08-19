import sys

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.cm import ScalarMappable
import matplotlib.pyplot as plt
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

from ._constants import *

class PortSelection(QObject):

    def __init__(self) -> None:
        """
        Port selection widget
        """
        super(PortSelection, self).__init__()

        self.group = QGroupBox()
        self.label = QLabel("Port")

        self.port_line_edit = QLineEdit("")

        self.valid_label = QLabel()
        self.set_invalid()

        # create widget layout
        self.layout = QHBoxLayout()
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.port_line_edit)
        self.layout.addWidget(self.valid_label)
        self.group.setLayout(self.layout)

    def set_valid(self):
        self.valid_label.setText("✓")
        self.valid_label.setStyleSheet("QLabel { color : green }")

    def set_invalid(self):
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
        filepath = QFileDialog.getSaveFileName(self, "Save Settings", filepath, "Json (*.json)")
        return filepath[0]
    def open_file(self, filepath):
        filepath = QFileDialog.getOpenFileName(self, "Open Settings", filepath, "Json (*.json)")
        return filepath[0]

class SaveSettingsWidget(QObject):
    changed_options = Signal()

    def __init__(self) -> None:
        super(SaveSettingsWidget, self).__init__()

        self.layout = QFormLayout()
        self.group = QGroupBox("Save Settings")
        self.group.setLayout(self.layout)

        self.filepath = QLineEdit()
        self.save = QPushButton("Save")

        self.layout.addRow(self.filepath, self.save)

        self.open = QPushButton("Open")
        self.open.setMinimumWidth(110)
        self.save_as = QPushButton("Save as")
        self.layout.addRow(self.open, self.save_as)

class FlimParamsWidget(QObject):

    changed = Signal(FlimParams)

    def __init__(self) -> None:
        super(FlimParamsWidget, self).__init__()

        self.group = QGroupBox("FLIM Parameters")
        self.layout = QFormLayout()
        self.group.setLayout(self.layout)

        def changed_callback():
            self.fit_start_label.setText("Fit Start= {:.2f}ns".format(self.fit_start.value() * self.period.value()))
            self.fit_end_label.setText("Fit End= {:.2f}ns".format(self.fit_end.value() * self.period.value()))
            self.changed.emit(self.values())

        self.period = QDoubleSpinBox()
        self.period.setMinimum(sys.float_info.min)
        self.period.setMaximum(MAX_VALUE)
        self.period.setSingleStep(0.01)
        self.period.valueChanged.connect(changed_callback)
        self.layout.addRow("Period (ns)", self.period)

        self.fit_start = QSpinBox()
        self.fit_start_label = QLabel("Fit Start")
        self.fit_start.setRange(0, 1)
        self.fit_start.valueChanged.connect(changed_callback)
        self.fit_start.valueChanged.connect(lambda start : self.fit_end.setMinimum(start + 1))
        self.layout.addRow(self.fit_start_label, self.fit_start)

        self.fit_end = QSpinBox()
        self.fit_end_label = QLabel("Fit End")
        self.fit_end.setRange(1, MAX_VALUE)
        self.fit_end.valueChanged.connect(changed_callback)
        self.fit_end.valueChanged.connect(lambda end : self.fit_start.setMaximum(end - 1))
        self.layout.addRow(self.fit_end_label, self.fit_end)

    def values(self):
        return FlimParams(
            self.period.value(),
            self.fit_start.value(),
            self.fit_end.value(),
        )

    def setValues(self, flim_params : FlimParams):
        self.period.setValue(flim_params.period)
        if flim_params.fit_start > self.fit_start.value():
            self.fit_end.setValue(flim_params.fit_end)
            self.fit_start.setValue(flim_params.fit_start)
        else:
            self.fit_start.setValue(flim_params.fit_start)
            self.fit_end.setValue(flim_params.fit_end)
            

class DisplaySettingsWidget(QObject):

    changed = Signal(DisplaySettings)

    def __init__(self) -> None:
        super(DisplaySettingsWidget, self).__init__()

        self.group = QGroupBox("Display Filters")
        self.layout = QFormLayout()
        self.group.setLayout(self.layout)

        def changed_callback():
            self.changed.emit(self.values())
            self.update_colorbar()

        self.max_chisq = QDoubleSpinBox()
        self.max_chisq.setRange(0, MAX_VALUE)
        self.max_chisq.valueChanged.connect(changed_callback)
        self.layout.addRow("Max χ2", self.max_chisq)

        self.min_tau = QDoubleSpinBox()
        self.min_tau.setRange(0, DEFAULT_MAX_TAU)
        self.min_tau.setSingleStep(0.1)
        self.min_tau.setDecimals(3)
        self.min_tau.valueChanged.connect(changed_callback)
        self.min_tau.valueChanged.connect(lambda min : self.max_tau.setMinimum(min))
        self.layout.addRow("Min Lifetime (ns)", self.min_tau)

        self.max_tau = QDoubleSpinBox()
        self.max_tau.setRange(0, MAX_VALUE)
        self.max_tau.setSingleStep(0.1)
        self.max_tau.setDecimals(3)
        self.max_tau.valueChanged.connect(changed_callback)
        self.max_tau.valueChanged.connect(lambda max : self.min_tau.setMaximum(max))
        self.layout.addRow("Max Lifetime (ns)", self.max_tau)

        self.colormap = QComboBox()
        self.colormap.addItems(COLORMAPS.keys())
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
        return DisplaySettings(
            self.max_chisq.value(),
            self.min_tau.value(),
            self.max_tau.value(),
            self.colormap.currentText(),
        )

    def setValues(self, display_filters : DisplaySettings):
        self.max_chisq.setValue(display_filters.max_chisq)
        if display_filters.min_tau > self.min_tau.value():
            self.max_tau.setValue(display_filters.max_tau)
            self.min_tau.setValue(display_filters.min_tau)
        else:
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
        