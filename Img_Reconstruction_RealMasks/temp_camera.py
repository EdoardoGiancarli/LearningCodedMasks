"""
Temporary test script for `CodedMaskCamera` and `codedmask()`.
"""

from pathlib import Path
from dataclasses import dataclass
from functools import cached_property

import numpy as np
import numpy.typing as npt
from scipy.signal import correlate

from mbloodmoon.mask import _bisect_interval
from mbloodmoon.mask import _fold
from mbloodmoon.images import _enlarge
from mbloodmoon.types import BinsRectangular, UpscaleFactor
from mbloodmoon.io import MaskDataLoader


@dataclass(frozen=True)
class CodedMaskCamera:
    """
    Dataclass containing a coded mask camera system.

    Handles mask pattern, detector geometry, and related calculations for coded mask imaging.

    Args:
        mdl: Mask data loader object containing mask and detector specifications.
        upscale_f: Tuple of upscaling factors for x and y dimensions.
    """

    mdl: MaskDataLoader
    upscale_f: UpscaleFactor

    def _bins_mask(
        self,
        upscale_f: UpscaleFactor,
    ) -> BinsRectangular:
        """Generates mask binning structure with given upscale factors."""
        def bins(
            start: float,
            stop: float,
            px_size: float,
            upscaling: int,
        ) -> npt.NDArray:
            """Defines binning structure."""
            return np.linspace(start, stop, int((stop - start) * upscaling / px_size) + 1)
        
        l, r = self.mdl["mask_minx"], self.mdl["mask_maxx"]
        b, t = self.mdl["mask_miny"], self.mdl["mask_maxy"]        
        return BinsRectangular(
            bins(l, r, self.mdl["mask_deltax"], upscale_f.x),
            bins(b, t, self.mdl["mask_deltay"], upscale_f.y),
        )
    
    def _bins_detector(
        self,
        upscale_f: UpscaleFactor,
    ) -> BinsRectangular:
        """Generates detector binning structure from mask with given upscale factors."""
        base_mask_bins = self._bins_mask(UpscaleFactor(1, 1))
        mask_bins = self.bins_mask
        xmin, xmax = _bisect_interval(base_mask_bins.x, self.mdl["detector_minx"], self.mdl["detector_maxx"])
        ymin, ymax = _bisect_interval(base_mask_bins.y, self.mdl["detector_miny"], self.mdl["detector_maxy"])
        return BinsRectangular(
            mask_bins.x[xmin * upscale_f.x : xmax * upscale_f.x + 1],
            mask_bins.y[ymin * upscale_f.y : ymax * upscale_f.y + 1],
        )
    
    def _bins_sky(self) -> BinsRectangular:
        """Generates binning structure for the reconstructed sky image."""
        m_bins, d_bins = self.bins_mask, self.bins_detector
        sy, sx = self.shape_sky
        xstep, ystep = (
            np.abs(m_bins.x[1] - m_bins.x[0]),
            np.abs(m_bins.y[1] - m_bins.y[0]),
        )
        return BinsRectangular(
            np.linspace(
                m_bins.x[0] + d_bins.x[0] + xstep / 2,
                m_bins.x[-1] + d_bins.x[-1] - xstep / 2,
                sx + 1,
            ),
            np.linspace(
                m_bins.y[0] + d_bins.y[0] + ystep / 2,
                m_bins.y[-1] + d_bins.y[-1] - ystep / 2,
                sy + 1,
            ),
        )

    @property
    def specs(self) -> dict:
        """Returns a dictionary of mask parameters useful for image reconstruction."""
        return self.mdl.specs
    
    @cached_property
    def bins_mask(self) -> BinsRectangular:
        """Binning structure for the mask pattern."""
        return self._bins_mask(self.upscale_f)

    @cached_property
    def bins_detector(self) -> BinsRectangular:
        """
        Binning structure for the detector.

         ◀────────────mask────────────▶
         │    │    │    │    │    │    │
         └────┴────┴────┴────┴────┴────┘
        -3   -2   -1    0    +1   +2   +3
              ┌─┬──┬────┬────┬──┬─┐
              │    │    │    │    │
                │               │
                ◀───detector────▶
                │               │
           detector_min   detector_max

        Notes:
            - the binning is taken directly from the (upscaled) mask binning structure, so
              that the detector binning and the mask binning are aligned.
            - note that the detector physical dimensions are not integer multiple of the
              pixels physical dimensions, as in the case for the mask.
            - the binning is structured to be self-consistent at different upscaling, i.e.
              the edges are fixed to the same values. In this way, for a given upscale
              factor `f` the detector shape is `f` times the basic detector one.
            - from above, it follows that the binning edges are not superimposed with respect
              to the physical detector edges (this is addressed in `self.bulk`).
        """
        return self._bins_detector(self.upscale_f)
    
    @cached_property
    def bins_sky(self) -> BinsRectangular:
        """
        Returns bins for the sky-shifts domain. While the , the sky-bins are not due to shape parity.

            │    │    │    │    │    │    │
           ◀────┴────┴──mask───┴────┴───▶┘
            0    1    2    3    4    5    6

                      │    │    │
                     ◀───det───▶
                      0    1    2

         │    │    │    │     │    │    │    │
        ◀────┴────┴────┴─sky─┴────┴────┴────▶
         0    1    2    3     4    5    6    7

        Notes:
            - mask and detector bins are aligned and refer to the binning edges.
            - the sky bins are not aligned to the mask and detector ones due to
              shape parity, and refer to the centers of the sky image pixels (e.g.,
              the shifts [0, 0] refers to the center of the sky-image central pixel).
        """
        return self._bins_sky()
    
    @cached_property
    def shape_mask(self) -> tuple[int, int]:
        """Shape of the mask array (rows, columns)."""
        bins = self.bins_mask
        return len(bins.y) - 1, len(bins.x) - 1
    
    @cached_property
    def shape_detector(self) -> tuple[int, int]:
        """Shape of the detector array (rows, columns)."""
        bins = self.bins_detector
        return len(bins.y) - 1, len(bins.x) - 1
    
    @cached_property
    def shape_sky(self) -> tuple[int, int]:
        """Shape of the reconstructed sky image (rows, columns)."""
        n, m = self.shape_mask
        u, v = self.shape_detector
        return n + u - 1, m + v - 1
    
    @cached_property
    def mask(self) -> npt.NDArray:
        """2D array representing the coded mask pattern."""
        base = _fold(self.mdl.mask, self._bins_mask(UpscaleFactor(1, 1))).astype(int)
        return _enlarge(base, self.upscale_f)

    @cached_property
    def decoder(self) -> npt.NDArray:
        """2D array representing the mask pattern used for decoding."""
        base = _fold(self.mdl.decoder, self._bins_mask(UpscaleFactor(1, 1)))
        return _enlarge(base, self.upscale_f)

    @cached_property
    def bulk(self) -> npt.NDArray:
        """
        2D array representing the bulk (sensitivity) array of the mask.

        Notes:
            - because the binning edges and the physical detector are not
              superimposed (see `self.bins_detector`), at a given upscaling
              `f` the redundant pixels which not correspond to a sensitive
              detector physical zone are set to zero.
        """
        base_bins = self._bins_mask(UpscaleFactor(1, 1))
        framed_bulk = _fold(self.mdl.bulk, base_bins)
        framed_bulk[~np.isclose(framed_bulk, np.zeros_like(framed_bulk))] = 1

        xmin, xmax = _bisect_interval(base_bins.x, self.mdl["detector_minx"], self.mdl["detector_maxx"])
        ymin, ymax = _bisect_interval(base_bins.y, self.mdl["detector_miny"], self.mdl["detector_maxy"])
        detector = _enlarge(framed_bulk[ymin : ymax, xmin : xmax], self.upscale_f)

        n_zero_resp_pxs_x = int(
            np.abs(
                (base_bins.x[xmin] - self.mdl["detector_minx"]) * self.upscale_f.x / self.mdl["mask_deltax"]
            )
        )
        n_zero_resp_pxs_y = int(
            np.abs(
                (base_bins.y[ymin] - self.mdl["detector_miny"]) * self.upscale_f.y / self.mdl["mask_deltay"]
            )
        )
        if n_zero_resp_pxs_x > 0:
            detector[:, :n_zero_resp_pxs_x] = 0
            detector[:, -n_zero_resp_pxs_x:] = 0
        if n_zero_resp_pxs_y > 0:
            detector[:n_zero_resp_pxs_y, :] = 0
            detector[-n_zero_resp_pxs_y:, :] = 0
        
        return detector

    @cached_property
    def balancing(self) -> npt.NDArray:
        """2D array representing the correlation between decoder and bulk patterns."""
        return correlate(self.decoder, self.bulk, mode="full")


def codedmask(
    mask_filepath: str | Path,
    upscale_x: int = 1,
    upscale_y: int = 1,
) -> CodedMaskCamera:
    """
    An interface to CodedMaskCamera.

    Args:
        mask_filepath: a str or a path object pointing to the mask filepath.
        upscale_x: upscaling factor over the x direction.
        upscale_y: upscaling factor over the y direction.

    Returns:
        a CodedMaskCamera object.

    Raises:
        ValueError: if physical detector plane is larger than mask.
        ValueError: if upscale factors are not positive integers.
    """
    mdl = MaskDataLoader(mask_filepath)

    if not (
        # fmt: off
        mdl["detector_minx"] >= mdl["mask_minx"] and
        mdl["detector_maxx"] <= mdl["mask_maxx"] and
        mdl["detector_miny"] >= mdl["mask_miny"] and
        mdl["detector_maxy"] <= mdl["mask_maxy"]
        # fmt: on
    ):
        raise ValueError("Detector plane is larger than mask.")

    if not (
        (isinstance(upscale_x, int) and upscale_x > 0) and
        (isinstance(upscale_y, int) and upscale_y > 0)
    ):
        raise ValueError("Upscale factors must be positive integers.")

    return CodedMaskCamera(mdl, UpscaleFactor(x=upscale_x, y=upscale_y))



# def _bins_detector(
#     self,
#     upscale_f: UpscaleFactor,
# ) -> BinsRectangular:
#     """Generate binning structure for detector with given upscale factors."""
#     mask_bins = self._bins_mask(upscale_f)
#     xmin, xmax = _bisect_interval(mask_bins.x, self.mdl["detector_minx"], self.mdl["detector_maxx"])
#     ymin, ymax = _bisect_interval(mask_bins.y, self.mdl["detector_miny"], self.mdl["detector_maxy"])
#     return BinsRectangular(
#         mask_bins.x[xmin : xmax + 1],
#         mask_bins.y[ymin : ymax + 1],
#     )


# @cached_property
# def bulk(self) -> npt.NDArray:
#     """2D array representing the bulk (sensitivity) array of the mask."""
#     framed_bulk = _fold(self.mdl.bulk, self._bins_mask(UpscaleFactor(1, 1)))
#     framed_bulk[~np.isclose(framed_bulk, np.zeros_like(framed_bulk))] = 1
#     bins = self._bins_mask(self.upscale_f)
#     xmin, xmax = _bisect_interval(bins.x, self.mdl["detector_minx"], self.mdl["detector_maxx"])
#     ymin, ymax = _bisect_interval(bins.y, self.mdl["detector_miny"], self.mdl["detector_maxy"])
#     return _enlarge(framed_bulk, self.upscale_f)[ymin : ymax, xmin : xmax]


# end