"""
Module with LEM-X coded-mask cameras angular resolution following Skinner et al, 2008.
"""

import numpy as np

from bloodmoon.coords import angle2shift
from bloodmoon.mask import CodedMaskCamera


def camera_angular_resolution(camera: CodedMaskCamera) -> tuple[float, float]:
    """
    Computes the camera angular resolution along the axes, in [arcmin].

    Args:
        camera (CodedMaskCamera):
            Instance with info on the camera system geometry.
    
    Returns:
        output (tuple[float, float]):
            Camera angular resolution along the (x, y) axes in [arcmin].
    
    ## Notes:
        * From: Skinner, G.K., 2008. Sensitivity of coded mask telescopes.
          Applied optics, 47(15), pp.2739-2749.
    """
    def angular_resolution(m_pitch: float, d_pitch: float, dist: float) -> float:
        """
        Computes the camera angular resolution along the axis, in [arcmin].

        Args:
            m_pitch (float): Mask element pitch.
            d_pitch (float): Detector element resolution pitch.
            dist (float): Mask - Detector distance.
        """
        dtheta_rad = np.sqrt(
            np.square(m_pitch / dist) + np.square(d_pitch / dist)
        )
        dtheta_arcmin = np.rad2deg(dtheta_rad) * 60
        return dtheta_arcmin
    
    p = camera.specs['mask_detector_distance']
    mx, my = (
        camera.specs['slit_deltax'],
        camera.specs['slit_deltay'],
    )
    dx, dy = (
        camera.specs['...'],
        camera.specs['...'],
    )
    return (
        angular_resolution(mx, dx, p),
        angular_resolution(my, dy, p),
    )


def camera_skycoords_errors(camera: CodedMaskCamera) -> tuple[float, float]:
    """
    Computes the camera local-frame coords NOMINAL* errors along
    the axes, taking into account the chosen camera upscaling.

    *For now, we are considering a proxy for the camera angular
     resolution along the fine and coarse directions.

    Args:
        camera (CodedMaskCamera):
            Instance with info on the camera system geometry.
    
    Returns:
        output (tuple[float, float]):
            Camera local-frame cartesian coords errors along
            the (x, y) axes in [mm].
    """
    def arcmin2deg(angle: float) -> float:
        """Converts angle from [arcmin] to [deg]."""
        return angle / 60
    
    UPX, UPY = camera.upscale_f
    ang_res_x, ang_res_y = camera_angular_resolution(camera)
    dsx = abs(angle2shift(camera, arcmin2deg(ang_res_x / UPX)))  # [mm]
    dsy = abs(angle2shift(camera, arcmin2deg(ang_res_y / UPY)))  # [mm]
    return dsx, dsy


# end