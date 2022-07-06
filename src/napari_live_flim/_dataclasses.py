from dataclasses import dataclass
from typing import TYPE_CHECKING
from numpy import ndarray
from flimlib import TripleIntegralResult, MarquardtResult

@dataclass(frozen=True)
class DisplayFilters():
    min_intensity : int
    max_chisq : float
    max_tau : float

@dataclass(frozen=True)
class FlimParams():
    period : float
    fit_start : int
    fit_end : int
    delta_snapshots : bool

@dataclass(frozen=True)
class SelectionResult:
    histogram : ndarray
    points : ndarray
    rld : TripleIntegralResult
    lm : MarquardtResult

@dataclass(frozen=True)
class MaskResult:
    extrema : ndarray
    mask : ndarray

@dataclass(frozen=True)
class ElementData:
    series_no : int
    seqno : int
    frame : ndarray

@dataclass(frozen=True)
class SeriesMetadata:
    series_no : int
    port : int
    shape : tuple
