"""
Images simulation for models testing.
"""

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from matplotlib.patches import RegularPolygon, Circle
import matplotlib.pyplot as plt
from tqdm import tqdm

FACECOLOR: str = 'white'


def draw_polygon(
    pos: tuple[int, int],
    numVertices: int,
    radius: float,
    rot_angle: float,
    alpha: float,
) -> RegularPolygon:
    """Pentagon representation."""
    kwargs = {
        'radius': radius,
        'orientation': np.deg2rad(rot_angle),
        'alpha': alpha,
        'facecolor': FACECOLOR,
    }
    return RegularPolygon(pos, numVertices, **kwargs)

def draw_circle(
    pos: tuple[int, int],
    radius: float,
    alpha: float,
) -> Circle:
    """Circle representation."""
    kwargs = {
        'alpha': alpha,
        'facecolor': FACECOLOR,
    }
    return Circle(pos, radius, **kwargs)

def plot_shape(
    arr: NDArray,
    shape: RegularPolygon | Circle,
    dpi: int | float,
    save_to: str | Path,
) -> None:
    """Draw given polygon on input arr."""
    fig, ax = plt.subplots(figsize=(1, 1))
    ax.matshow(arr, cmap='gray', vmin=0, vmax=1)
    ax.add_patch(shape)
    ax.axis('off')
    plt.title('')
    plt.savefig(save_to, dpi=dpi, bbox_inches='tight', pad_inches=0.0)
    plt.close()
    return None

def randomize_shape(n: int, arr_dim: int) -> dict[str, NDArray]:
    """Initialises `n` random values for polygon."""
    vals =  {
        'pos': np.random.randint(int(0.25 * arr_dim), int(0.75 * arr_dim), (n, 2)),
        'radius': np.random.randint(int(0.125 * arr_dim), int(0.4 * arr_dim), n),
        'rot_angle': np.random.randint(0, 45, n),
        'alpha': 0.9 * np.ones(n),
    }
    return vals

def simulate_imgs(
    num_imgs: int,
    polygonVertices: int,
    save_to_dir: str | Path,
    start_num: int = 0,
    dpi: int | float = 120,
    frmt: str = 'png',
    arr_dim: int = 128,
) -> None:
    """Image simulation."""
    SHAPES_MAP = {
        -1: 'circle', 3: 'triangle', 4: 'square',
        5: 'pentagon', 6: 'hexagon', 7: 'heptagon',
        8: 'octagon', 9: 'nonagon', 10: 'decagon',
    }
    if polygonVertices not in list(SHAPES_MAP.keys()):
        raise ValueError(f"Invalid polygon vertices '{polygonVertices}'. Supported polygons are: {list(SHAPES_MAP.values())}.")
    
    arr = np.zeros((arr_dim, arr_dim))
    vals = randomize_shape(num_imgs, arr_dim)

    if polygonVertices == -1:
        draw_shape = lambda vals, idx: draw_circle(vals['pos'][idx], vals['radius'][idx], vals['alpha'][idx])
    else:
        draw_shape = lambda vals, idx: draw_polygon(
            vals['pos'][idx], polygonVertices, vals['radius'][idx], vals['rot_angle'][idx], vals['alpha'][idx],
        )
    
    for n in tqdm(range(num_imgs)):
        shape = draw_shape(vals, n)
        figname = f'{SHAPES_MAP[polygonVertices]}{n + start_num}'
        plot_shape(arr, shape, dpi, f'{save_to_dir}/{figname}.{frmt}')
    
    return




def main() -> None:
    DIRPATH: str = '/home/edoardo/Desktop/MockDataForDMs/ImgsMockDatasetDMs'
    # DIRPATH: str = '/mnt/d/MockDataForDMs/ImgsMockDatasetDMs'

    GEN_DATASET: bool = False

    if GEN_DATASET:
        num_imgs: int = 1000
        polygonVertices: tuple = (-1, 3, 4, 6)

        for pv in polygonVertices:
            simulate_imgs(
                num_imgs=num_imgs,
                polygonVertices=pv,
                save_to_dir=DIRPATH,
                start_num=0,
                dpi=110,
            )
    else:
        print('Dataset already generated!')
    return


if __name__ == '__main__':
    main()


# end