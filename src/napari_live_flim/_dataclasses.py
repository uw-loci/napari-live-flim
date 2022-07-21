from dataclasses import dataclass
import numpy as np
from flimlib import TripleIntegralResult, MarquardtResult

@dataclass(frozen=True)
class DisplayFilters():
    max_chisq : float
    min_tau : float
    max_tau : float
    colormap : str

@dataclass(frozen=True)
class FlimParams():
    period : float
    fit_start : int
    fit_end : int

@dataclass(frozen=True)
class SelectionResult:
    histogram : np.ndarray
    points : np.ndarray
    rld : TripleIntegralResult
    lm : MarquardtResult

@dataclass(frozen=True)
class MaskResult:
    extrema : np.ndarray
    mask : np.ndarray

@dataclass(init=False) # cannot be frozen because overridden __init__
class ElementData:
    series_no : int
    seqno : int
    frame : np.ndarray
    def __init__(self, series_no : int, seqno : int, frame : np.ndarray):
        self.series_no = series_no
        self.seqno = seqno
        self.frame = np.array(frame, dtype=np.float32, copy=True)

@dataclass(frozen=True)
class SeriesMetadata:
    series_no : int
    port : int
    shape : tuple
