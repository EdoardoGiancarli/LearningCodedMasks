"""
Data filters for photons energy range, sources flux and sources positions.
"""

from collections.abc import Sequence, Callable

import numpy.typing as npt
import numpy as np


def data_filter(
    record: np.recarray,
    energy_range: int | float | tuple[int | float, int | float] | None,
    coords: tuple[float, float] | Sequence[tuple[float, float]] | None,
) -> np.recarray:
    """
    Filters the input `record` based on the photons energy and/or position.
    
    Args:
        record (np.recarray): Input simulated data container.
        energy_range (int | float | tuple[int | float, int | float] | None):
            Energy range in keV for the data filtering. If a specific energy
            is given, this will be considered as the maximum filter value.
            If a tuple is given, it's interpreted as (`E_min`, `E_max`).
        coords (tuple[float, float] | Sequence[tuple[float, float]] | None):
            Input photons RA/Dec (or sequence of RA/Dec) to filter out.
    
    Returns:
        output (np.recarray): Output filtered data container.
    """
    def _energy_mask(
        mask: npt.NDArray,
        values: int | float | tuple[int | float, int | float],
    ) -> npt.NDArray:
        """Creates an energy mask for the input `record`."""
        if isinstance(values, (int, float)):
            mask &= (record["ENERGY"] > values)
        else:
            mask &= (record["ENERGY"] > values[0]) & (record["ENERGY"] < values[1])
        return mask

    def _coords_mask(
        mask: npt.NDArray,
        values: tuple[float, float],
    ) -> npt.NDArray:
        """Creates a RA/Dec mask for the input `record`."""
        # to address float64 to float32 conv, we remove
        # the photons coming from the specified RA/Dec
        mask &= ~(
            (np.abs(record["RA"] - values[0]) < 1e-7) &
            (np.abs(record["DEC"] - values[1]) < 1e-7)
        )
        #mask &= (record["RA"] != values[0]) | (record["DEC"] != values[1])
        return mask

    mask = np.ones(len(record), dtype=bool)

    if energy_range is not None:
        mask = _energy_mask(mask, energy_range)
    
    if coords is not None:
        if isinstance(coords[0], float):
            mask = _coords_mask(mask, coords)
        else:
            _cmask = np.ones(len(record), dtype=bool)
            for c in coords:
                _cmask = _coords_mask(_cmask, c)
            mask &= _cmask
    
    return record[mask]


def flux_filter(
    flux_range: int | float | tuple[int | float, int | float],
) -> Callable[[np.recarray], np.recarray]:
    """
    Filters the input catalog `record` for a given flux range.

    Args:
        flux_range (int | float | tuple[int | float, int | float]):
            Flux range in ph/cm2/s for the data filtering. If a specific flux
            is given, this will be considered as the minimum filter value.
            If a tuple is given, it's interpreted as (`F_min`, `F_max`).

    Returns:
        apply (Callable[[np.recarray], np.recarray]):
            Filter application to the given simulated photon list.
    """
    def apply(record: np.recarray) -> np.recarray:
        """
        Applies the filter in the specified flux range.

        Args:
            record (np.recarray): Input simulated data container.
        
        Returns:
            output (np.recarray): Output filtered data container.
        """
        if isinstance(flux_range, (int, float)):
            filtered = record[record["FLUX"] > flux_range]
        else:
            filtered = record[
                (record["FLUX"] > flux_range[0]) &
                (record["FLUX"] < flux_range[1])
            ]
        return filtered
    
    return apply


def source_filter(n: int | tuple[int, int]) -> Callable[[np.recarray], np.recarray]:
    """
    Select the `n` brightest sources from the input catalog `record`,
    or a given interval of sources.

    Args:
        n (int | tuple[int]):
            Filtered interval of sources, up to the n-th brightest
            source or from `n[0]` to `n[1]` if `n` is a tuple.

    Returns:
        apply (Callable[[np.recarray], np.recarray]):
            Filter application to the given simulated photon list.
    """
    def apply(record: np.recarray) -> np.recarray:
        """
        Applies the filter for the specified number of sources.

        Args:
            record (np.recarray): Input simulated data container.
        
        Returns:
            output (np.recarray): Output filtered data container.
        """
        sorted_record = np.sort(record, order="NPHOTONS")[::-1]
        return sorted_record[:n] if isinstance(n, int) else sorted_record[n[0] : n[1]]
    
    return apply


# end