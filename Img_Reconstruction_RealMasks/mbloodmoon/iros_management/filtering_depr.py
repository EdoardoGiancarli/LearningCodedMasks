"""
Simulated data filters for photons energy range, sources flux and sources positions.
"""

from collections.abc import Sequence, Callable
import numpy as np


def energy_filter(
    energy_range: int | float | tuple[int | float],
) -> Callable[[np.recarray], np.recarray]:
    """
    Filters the input `record` for a given energy range.

    Args:
        energy_range (int | float | tuple[int | float]):
            Energy range in keV for the data filtering. If a specific energy
            is given, this will be considered as the minimum filter value.
            If a tuple is given, it's interpreted as (`E_min`, `E_max`).

    Returns:
        apply (Callable[[np.recarray], np.recarray]):
            Filter application to the given simulated photon list.
    """
    def apply(record: np.recarray) -> np.recarray:
        """
        Applies the filter in the specified energy range.

        Args:
            record (np.recarray): Input simulated data container.
        
        Returns:
            output (np.recarray): Output filtered data container.
        """
        if isinstance(energy_range, (int, float)):
            filtered = record[record["ENERGY"] > energy_range]
        else:
            filtered = record[
                (record["ENERGY"] > energy_range[0]) &
                (record["ENERGY"] < energy_range[1])
            ]
        return filtered

    return apply


def position_filter(
    coords: tuple[float],
) -> Callable[[np.recarray], np.recarray]:
    """
    Excludes the photons in the input `record` originated from the input RA/Dec.

    Args:
        coords (tuple[float]):
            Input photons RA/Dec (or sequence of RA/Dec) to filter out.

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
        raise NotImplementedError
    
    return apply


def flux_filter(
    flux_range: int | float | tuple[int | float],
) -> Callable[[np.recarray], np.recarray]:
    """
    Filters the input `record` for a given flux range.

    Args:
        flux_range (int | float | tuple[int | float]):
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


def source_filter(n: int | tuple[int]) -> Callable[[np.recarray], np.recarray]:
    """
    Select the `n` brightest sources from the input `record`,
    or a given interval of sources.

    Args:
        n (int | tuple[int]):
            Filtered interval of sources, up to the n-th brightest
            source or from n[0] to n[1] if `n` is a tuple.

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
        raise NotImplementedError
    
    return apply


def configure_filters(
    energy_range: int | float | tuple[int | float] | None,
    coords: tuple[float] | Sequence[tuple[float]] | None,
) -> list[Callable[[np.recarray], np.recarray]]:
    """
    Configures the filter list that will be applied to the simulated photons list.

    Args:
        energy_range (int | float | tuple[int | float] | None):
            Energy range in keV for the data filtering. If a specific energy
            is given, this will be considered as the minimum filter value.
            If a tuple is given, it's interpreted as (`E_min`, `E_max`).
        coords (tuple[float] | Sequence[tuple[float]] | None):
            Input photons RA/Dec (or sequence of RA/Dec) to filter out.
    
    Returns:
        output (list[Callable]):
            List of filters to apply to the data.
    """
    filters = []

    for par, func in zip(
        (energy_range, coords),
        (energy_filter, position_filter),
    ):
        if par: filters.append(func(par))

    return filters


def apply_filters(
    filters: list[Callable[[np.recarray], np.recarray]],
    data: np.recarray,
) -> np.recarray:
    """
    Applies the choosen filters to the simulated photons list.

    Args:
        filters (list[Callable]): List of filters to apply.
        data (np.recarray): Simulated data container.
    """
    for func in filters:
        data = func(data)
    
    return data



"""
    IDEA:


    if np.any(energy_range, ...):
        list_filters = configure_filters()

    if list_filters: apply_filters(list_filters, sdl_data)
"""


# end