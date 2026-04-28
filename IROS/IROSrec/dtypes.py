"""
Module for the IROS sky-reconstruction pipeline datatypes.
"""

from dataclasses import dataclass
from typing import NamedTuple, Sequence

from bloodmoon.types import CoordEquatorial


@dataclass(frozen=True)
class AnalysisParams:
    """Pipeline data paths and analysis config params container."""
    # path to data (mask, WISEMAN events and directory to save output files)
    mask_file: str
    simul_data: str
    save_path: str
    # instrumental effects
    vignetting: bool
    psfy: bool
    # LEM-X module to apply IROS to, events reconstruction type and images sampling
    unit_camsID: tuple[str, str]
    dataset: str
    start_ups: tuple[int, int]
    final_ups: tuple[int, int]
    sky_compositions: bool


@dataclass(frozen=True)
class IROSParams:
    """Pipeline config values for IROS container."""
    iros_max_iterations: int
    iros_snr_threshold: int | float
    smoothing: bool
    smoothing_thresh: int | float | None
    smoothing_baseline_recnstr: str | None


@dataclass(frozen=True)
class OutFileNames:
    """Pipeline output databases, skies/snr, .reg and info filenames container."""
    unit_camsID: tuple[str, str]
    sim_sky: tuple[str, str]
    comp_sim_sky: str
    out_db: str
    iros_res: tuple[str, str]
    comp_iros_res: str
    srcs_db: str
    out_sky: tuple[str, str]
    comp_out_sky: str
    out_reg: tuple[str, str]
    pipeline_params: str


@dataclass(frozen=True)
class WMFilters:
    """Pipeline data filtering values container."""
    E_min: int | float | None
    E_max: int | float | None
    coords: CoordEquatorial | Sequence[CoordEquatorial] | None
    F_min: int | float | None
    F_max: int | float | None


class PipelineParams(NamedTuple):
    """
    Container for the IROS pipeline parameters, which configures the LEM-X coded-mask cameras
    module parameters, the IROS setup parameters, handles the pipeline output files setup
    (skyes, databases, info files), and manage the data photons energy range and catalogue
    fluxes range to perform the IROS sources candidates association with.
    """
    analysis_params: AnalysisParams
    iros_params: IROSParams
    filenames: OutFileNames
    filters: WMFilters


# end