'''
Testing optimisation procedure....perché non funzia :c
'''
from singleCAM_IROS._pipeline_support import _handle_dirpaths

from typing import Callable
from pathlib import Path

import numpy as np
from numpy.typing import NDArray 
from scipy.optimize import least_squares

from bloodmoon.coords import pos2shift
from bloodmoon.mask import CodedMaskCamera, codedmask
from bloodmoon.mask import count, decode
from bloodmoon.mask import variance, snratio
from bloodmoon.io import simulation_files
from bloodmoon.optim import model_sky

import darksun as ds


def handle_det_spres(dataset: str) -> bool:
    """Handles detector spatial resolution correction."""
    if dataset not in ('detected', 'reconstructed'):
        raise ValueError('Nah-huh...')
    
    return False if dataset == 'detected' else True


def find_candidate(
    sky: NDArray,
    snr: NDArray,
    snr_threshold: int | float,
    batch: int = 1000,
) -> tuple[int, int] | bool:
    """
    Returns the position of a valid IROS candidate inside the sky image.
    """
    reservoir = np.array(
        [np.unravel_index(id_, sky.shape) for id_ in np.argsort(sky, axis=None)[-batch:]]
    )
    for pos in reservoir[::-1]:
        if (snr[*pos] > snr_threshold):
            return tuple(pos)
    return False


def _init_source_model(
    camera: CodedMaskCamera,
    vignetting: bool = True,
    psfy: bool = True,
) -> Callable[[float, float, float], NDArray]:
    """
    A slow, vanilla implementation of the model for both direction and fluence optimization.
    Intended for debugging and benchmarking.

    Args:
        camera: CodedMaskCamera instance containing all geometric parameters
        vignetting: If true, shadowgram model simulates vignetting.
        psfy: If true, the model used for optimization will simulate detector position
        reconstruction effects.

    Returns:
        A Callable, which is the routine for computing the model.
    """

    def f(shift_x: float, shift_y: float, fluence: float) -> NDArray:
        """
        A simple, slow version of the model for both direction and fluence optimization.

        Args:
            shift_x: Source position x-coordinate in sky-shift space (mm)
            shift_y: Source position y-coordinate in sky-shift space (mm)
            fluence: Source intensity/fluence value

        Returns:
            2D array representing the modeled sky reconstruction
        """
        return model_sky(camera, shift_x, shift_y, fluence, vignetting=vignetting, psfy=psfy)
    
    return f


def _init_loss_metric(
    true: NDArray,
    pos: tuple[int, int],
    camera: CodedMaskCamera,
    model_source: Callable[[float, float, float], NDArray],
) -> Callable[
    [tuple[float, float, float]],
    float,
]:
    """
    Initialises the loss.
    """
    cropy, cropx = (
        int(camera.specs.slit_deltay * camera.upscale_f.y / camera.specs.mask_deltay) + 1,
        int(camera.specs.slit_deltax * camera.upscale_f.x / camera.specs.mask_deltax) + 1,
    )
    i, j = pos
    slicey, slicex = (
        slice(i - cropy, i + cropy + 1),
        slice(j - cropx, j + cropx + 1),
    )
    true_ = true[slicey, slicex]

    def f(args: tuple[float, float, float]) -> float:
        """Loss metric for optimisation."""
        sky = model_source(*args)[slicey, slicex]
        metric_val = np.mean(np.square(sky - true_))
        return metric_val
    
    return f


def optimise(
    true: NDArray,
    snrmap: NDArray,
    camera: CodedMaskCamera,
    vignetting: bool,
    psfy: bool,
    verbose: bool = True,
) -> tuple[float, float, float]:
    """
    Source parameters optimisation procedure.
    """
    px_dim_x, px_dim_y = (
        camera.specs.mask_deltax / camera.upscale_f.x,
        camera.specs.mask_deltay / camera.upscale_f.y,
    )

    source_model = _init_source_model(camera, vignetting, psfy)
    arg_sky = find_candidate(
        sky=true,
        snr=snrmap,
        snr_threshold=5.0,
    )
    loss = _init_loss_metric(true, arg_sky, camera, source_model)

    sx_start, sy_start = pos2shift(camera, *arg_sky)
    fluence_start = true[*arg_sky] / 0.9                 # camera coding power (Skinner et al. 2008)

    with ds.timer('Optimising'):
        results = least_squares(
            loss,
            x0=np.array((sx_start, sy_start, fluence_start)),
            bounds=[
                (
                    max(sx_start - 3 * px_dim_x, camera.bins_sky.x[0]),
                    max(sy_start - 3 * px_dim_y, camera.bins_sky.y[0]),
                    true[*arg_sky],
                ),
                (
                    min(sx_start + 3 * px_dim_x, camera.bins_sky.x[-1]),
                    min(sy_start + 3 * px_dim_y, camera.bins_sky.y[-1]),
                    true[*arg_sky] / 0.8,
                ),
            ],
            xtol=1e-7,
            ftol=1e-6,
            x_scale='jac',
        )
    # store the final optimized positions and fluence
    sx, sy, fluence = map(float, results.x[:3])

    # optimization verbose
    if verbose:
        print(
            f'\n'
            f'## Optimisation Results:\n'
            f'  - fluence START: {fluence_start}\n'
            f'  - shifts START (x, y): {sx_start}, {sy_start}\n'

            f'  - fluence OPTIM.: {fluence}\n'
            f'  - shifts OPTIM. (x, y): {sx}, {sy}\n'

            f'  - fluence GAIN %: {(fluence - fluence_start) * 100 / fluence_start:.3f}\n'
            f'  - shift_x GAIN %: {(sx - sx_start) * 100 / sx_start:.3f}\n'
            f'  - shift_y GAIN %: {(sy - sy_start) * 100 / sy_start:.3f}\n'
        )

    return sx, sy, fluence






def main():
    # --- sim data
    MASK_FITS: str = "wfm_mask_NTHT_20250725.fits"

    SKYFIELD: str = "GalacticCentre"
    DATA_FITS: str = "galctr_rxte-sax_2-50keV_mask_050_1040x17_opaquemask_infdet"

    ID_CAMERA_A: str = "cam1a"
    ID_CAMERA_B: str = "cam1b"
    DATASET: str = "reconstructed"

    UPS_X: int = 2
    UPS_Y: int = 1

    VIGNETTING: bool = True
    PSFY: bool = handle_det_spres(DATASET)

    # --- init data
    mask_path, simul_data, _ = _handle_dirpaths(
        mask=MASK_FITS,
        skyfield=SKYFIELD,
        simul=DATA_FITS,
    )
    wfm: CodedMaskCamera = codedmask(mask_path, UPS_X, UPS_Y)
    filepaths: dict[str, dict[str, Path]] = simulation_files(simul_data)
    sdlA = ds.get_data(filepaths[ID_CAMERA_A][DATASET])
    sdlB = ds.get_data(filepaths[ID_CAMERA_B][DATASET])

    # --- gen images
    detector_camA = count(wfm, sdlA.DLdata)[0]
    true_sky_camA = decode(wfm, detector_camA)
    varmap_camA = variance(wfm, detector_camA)
    snr_camA = snratio(true_sky_camA, varmap_camA)
    results_camA = optimise(true_sky_camA, snr_camA, wfm, VIGNETTING, PSFY)
    true_params_camA = (
        ...,
    )

    detector_camB = count(wfm, sdlB.DLdata)[0]
    true_sky_camB = decode(wfm, detector_camB)
    varmap_camB = variance(wfm, detector_camB)
    snr_camB = snratio(true_sky_camB, varmap_camB)
    results_camB = optimise(true_sky_camB, snr_camB, wfm, VIGNETTING, PSFY)
    true_params_camB = (
        ...,
    )

    print(
        f'Fit CAM1A: {results_camA}\n'
        f'Fit CAM1B: {results_camB}\n'
    )




if __name__ == '__main__':
    main()


# end