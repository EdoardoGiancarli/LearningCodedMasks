"""
Simulated data filters for photons energy range, sources flux and sources positions.
"""

from collections.abc import Sequence, Callable

import numpy as np

from mbloodmoon.io import SimulationDataLoader


def energy_filter(
    energy_range: int | float | tuple[int | float],
) -> Callable[[np.recarray], np.recarray]:
    """
    Filters the input `record` for a given energy range.

    Args:
        energy_range (int | float | tuple[int | float]):
            Energy range in keV for the data filtering. If a specific energy
            is given, this will be considered as the minimum filter value.

    Returns:
        apply (Callable):
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
        raise NotImplementedError
    
    return apply


def flux_filter(
    record: np.recarray,
    flux_range: int | float | tuple[int | float],
) -> np.recarray:
    """
    Filters the input `record` for a given flux range.

    Args:
        record (np.recarray):
            Input simulated data container.
        flux_range (int | float | tuple[int | float]):
            Flux range in ph/cm2/s for the data filtering. If a specific flux
            is given, this will be considered as the minimum filter value.

    Returns:
        output (np.recarray):
            Output data container in the specified flux range.
    """
    raise NotImplementedError


def source_filter(
    record: np.recarray,
    n: int | tuple[int],
) -> np.recarray:
    """
    Select the `n` brightest sources from the input `record`,
    or a given interval of sources.

    Args:
        record (np.recarray):
            Input simulated data container.
        n (int | tuple[int]):
            Filtered interval of sources, up to the n-th brightest
            source or from n[0] to n[1] if `n` is a tuple.

    Returns:
        output (np.recarray):
            Output data container with the selected sources.
    """
    raise NotImplementedError


def position_filter(
    record: np.recarray,
    coords: tuple[float] | Sequence[tuple[float]],
) -> np.recarray:
    """
    Excludes the photons in the input `record` originated from the input RA/Dec.

    Args:
        record (np.recarray):
            Input simulated data container.
        coords (tuple[float] | Sequence[tuple[float]]):
            Input photons RA/Dec (or sequence of RA/Dec) to filter out.

    Returns:
        output (np.recarray):
            Output data container without the photons from specified `coords`.
    """
    raise NotImplementedError


def configure_filters(
    energy_range: int | float | tuple[int | float],
    flux_range: int | float | tuple[int | float],
    n: int | tuple[int],
    coords: tuple[float] | Sequence[tuple[float]],
) -> list[Callable[[np.recarray], np.recarray]]:
    """
    Configures the filter list that will be applied to the simulated photons list.

    Args:
        energy_range (int | float | tuple[int | float]):
            Energy range in keV for the data filtering. If a specific energy
            is given, this will be considered as the minimum filter value.
        flux_range (int | float | tuple[int | float]):
            Flux range in ph/cm2/s for the data filtering. If a specific flux
            is given, this will be considered as the minimum filter value.
        n (int | tuple[int]):
            Filtered interval of sources, up to the n-th brightest
            source or from n[0] to n[1] if `n` is a tuple.
        coords (tuple[float] | Sequence[tuple[float]]):
            Input photons RA/Dec (or sequence of RA/Dec) to filter out.
    
    Returns:
        output (list[Callable]):
            List of filters (if any) to apply to the data.
    """
    filters = []

    for par, func in zip(
        (energy_range, flux_range, n, coords),
        (energy_filter, flux_filter, source_filter, position_filter),
    ):
        if par: filters.append(func(par))

    return filters


def apply_filters(
    filters: list[Callable[[np.recarray], np.recarray]],
    sdl_data: np.recarray,
) -> np.recarray:
    """
    Applies the choosen filters to the simulated photons list.

    Args:
        filters (list[Callable]): List of filters to apply.
        sdl_data (np.recarray): Simulated data container.
    """
    for func in filters:
        sdl_data = func(sdl_data)
    
    return sdl_data



"""
    IDEA:


    if np.any(energy_range, ...):
        list_filters = configure_filters()

    if list_filters: apply_filters(list_filters, sdl_data)
"""


# end