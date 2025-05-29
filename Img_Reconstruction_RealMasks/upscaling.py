import numpy as np
import numpy.typing as npt

from mbloodmoon.types import UpscaleFactor


def _enlarge(
    m: npt.NDArray,
    upscale_f: UpscaleFactor,
) -> npt.NDArray:
    """
    Oversamples a 2D array by repeating elements along the axes.

    Args:
        m (npt.NDArray): Input 2D array.
        upscale_f (UpscaleFactor): Upscaling factors.

    Returns:
        output (npt.NDArray): Oversampled array.

    Notes:
        - the total sum is NOT conserved.
    """    
    for i, f in enumerate(upscale_f[::-1]):
        m = np.repeat(m, f, axis=i)
    return m


def upscale(
    data: npt.NDArray,
    upscale_y: int = 1,
    upscale_x: int = 1,
) -> npt.NDArray:
    """
    Upscales a 2D array by repeating elements along each axis and
    by interpolating array values.

    Args:
        data (npt.NDArray): Input 2D array.
        upscale_y (int): Upscaling factor over the y direction.
        upscale_x (int): Upscaling factor over the x direction.

    Returns:
        output (npt.NDArray): Oversampled array.

    Raises:
        ValueError: if upscale factors are not positive integers.
    
    Notes:
        - The array total sum is conserved through linear interpolation.
        - For N-dim arrays, consider using `astropy.nndata.block_replicate()`.
    """
    if not (
        (isinstance(upscale_y, int) and upscale_y > 0) and
        (isinstance(upscale_x, int) and upscale_x > 0)
    ):
        raise ValueError("Upscaling factors must be positive integers.")
    
    upscaling = UpscaleFactor(upscale_x, upscale_y)
    return _enlarge(data, upscaling) / np.prod(upscaling)


def _reduce(
    m: npt.NDArray,
    downscaling: npt.NDArray,
) -> npt.NDArray:
    """
    Downsamples a 2D array.

    Args:
        m (npt.NDArray): Input 2D array.
        downscaling (npt.NDArray): Downscaling factors.

    Returns:
        output (npt.NDArray): Downsampled array.

    Notes:
        - the total sum is conserved.
    """
    def _handle_shape(
        data: npt.NDArray,
        factors: npt.NDArray,
    ) -> npt.NDArray:
        """Adjusts array for blocks subdivision by cutting extra-rows/columns."""

        def _handle_axis(a: npt.NDArray, idx: int) -> npt.NDArray:
            """Redistributes cutted values in the block-adjusted axis."""
            return a[:idx] + a[idx:].sum(axis=0) / idx
        
        adj_shape = (np.array(data.shape) // factors) * factors
        for ax in range(data.ndim):
            if data.shape[ax] != adj_shape[ax]:
                data = data.swapaxes(0, ax)
                data = _handle_axis(data, adj_shape[ax])
                data = data.swapaxes(0, ax)
        return data

    def _to_blocks(
        data: npt.NDArray,
        factors: npt.NDArray,
    ) -> npt.NDArray:
        """Reshapes input array into blocks."""
        assert not np.any(np.mod(data.shape, factors) != 0)
        nblocks = np.array(data.shape) // factors
        reshaping = tuple(dim for dims in zip(nblocks, factors) for dim in dims)
        return data.reshape(reshaping).transpose((0, 2, 1, 3))
    
    m = _handle_shape(m, downscaling)
    m = _to_blocks(m, downscaling)
    return m.sum(axis=(2, 3))


def downscale(
    data: npt.NDArray,
    downscale_y: int = 1,
    downscale_x: int = 1,
) -> npt.NDArray:
    """
    Downscales a 2D array by dividing the input array in blocks
    and adding over them to interpolate array values.

    Args:
        data (npt.NDArray): Input 2D array.
        downscale_y (int): Downscaling factor over the y direction.
        downscale_x (int): Downscaling factor over the x direction.

    Returns:
        output (npt.NDArray): Downsampled array.

    Raises:
        ValueError: if downscale factors are not positive integers.
    
    Notes:
        - The downsampling is performed through blocks subdivision, which
          represent the elements of the downsampled array. Each block is
          reduced by adding its elements for linear interpolation.
        - The total sum of the array is conserved.
        - For N-dim arrays, consider using `astropy.nndata.block_reduce()`.
    """
    
    if not (
        (isinstance(downscale_y, int) and downscale_y > 0) and
        (isinstance(downscale_x, int) and downscale_x > 0)
    ):
        raise ValueError("Downscaling factors must be positive integers.")
    
    downscaling = np.array((downscale_y, downscale_x))
    return _reduce(data, downscaling)