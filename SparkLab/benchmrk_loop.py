"""
Module for benchmarking training loops.
"""

from itertools import islice, cycle
from pathlib import Path
from PIL import Image
import math
import time
from typing import Any, Callable

from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from torch.types import Tensor
from torchvision.transforms import Compose

import spark as sp
from spark.handle import get_data_filespaths


def benchmark_func(
    func: Callable[[Any], Any],
    *args: Any,
    iterations: int = 500,
) -> tuple[float, float, Any]:
    """
    Benchmarks input `func` by running it for a specified number of times.
    The benchmark is performed by first calling the function to account
    for JIT compilation, caching and first-call effects (not included in
    the final performance time computation).

    Args:
        func (Callable[[Any], Any]):
            Function to benchmark.
        args (Any):
            Input `func` arguments.
        iterations (int, optional (default=`500`)):
            Number of call repetitions.
    
    Returns:
        output (tuple[float, float, Any]):
            - (float): Benchmarking repetitions averaged time.
            - (float): Averaged time error.
            - (Any): Input `func` results.
    """
    func(*args)
    result = None
    rep_time = []

    for _ in range(iterations):
        start_time = time.perf_counter()
        result = func(*args)
        end_time = time.perf_counter()

        delta = end_time - start_time
        rep_time.append(delta)

    rep_time_ = torch.tensor(rep_time)
    average, error = rep_time_.mean(dim=0), rep_time_.std(dim=0)

    return average.item(), error.item(), result



class ImageDataset(Dataset):
    """
    Baseline AutoEncoder dataset configurator.
    """
    def __init__(self, data_path: str | Path, transform: Compose | None) -> None:
        files_list = get_data_filespaths(data_path, shuffle=True)
        self.data = sp.process_data(files_list, lambda img: Image.open(img).convert('L'), transform)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, index) -> tuple[Tensor, Tensor]:
        sample = self.data[index]
        return sample, sample



def train_loop_separated(train_dl: DataLoader, valid_dl: DataLoader, epochs: int) -> None:
    """Trains and validates the model by looping per epoch."""
    for epoch in tqdm(range(epochs)):

        for x_batch, trg_batch in train_dl:
            time.sleep(1e-2)
        
        for x_batch, trg_batch in valid_dl:
            pass
    
    return None

def train_loop_itertools(train_dl: DataLoader, valid_dl: DataLoader, epochs: int) -> None:
    """Trains and validates the model in a `islice` + `cycle` loop logic."""
    nbatches: int = len(train_dl)
    total_steps: int = epochs * nbatches
    loop = tqdm(enumerate(islice(cycle(train_dl), total_steps)))

    for idx, (input_batch, trg_batch) in loop:
        info: dict = {'epoch': f'{idx // nbatches + 1}/{epochs}'}
        loop.set_postfix(info)
        time.sleep(1e-2)

        if (idx + 1) % nbatches == 0:
            for x_batch, trg_batch in valid_dl:
                pass
    
    return None

def train_loop_hybrid(train_dl: DataLoader, valid_dl: DataLoader, epochs: int) -> None:
    """Trains and validates the model by looping per epoch."""
    for epoch in tqdm(range(epochs)):

        for x_batch, trg_batch in islice(train_dl, len(train_dl)):
            time.sleep(1e-2)
        
        for x_batch, trg_batch in islice(valid_dl, len(valid_dl)):
            pass
    
    return None




def main() -> None:
    #BASEPATH: str = '/home/edoardo/Desktop/MockDataForDMs'
    BASEPATH: str = '/mnt/d/MockDataForDMs'
    DATASET_ID: str = 'mockPolyImgsDataset'

    EPOCHS: int = 100
    BATCH_SIZE: int = 100
    VALID_SIZE: int = 0.2

    dataset: Dataset = sp.load_dataset(f'{BASEPATH}/{DATASET_ID}.pt')
    train_dl, valid_dl = sp.get_dataloaders(dataset, BATCH_SIZE, VALID_SIZE)

    ITERATIONS: int = 3
    # b1 = benchmark_func(train_loop_separated, train_dl, valid_dl, EPOCHS, iterations=ITERATIONS)
    # b2 = benchmark_func(train_loop_itertools, train_dl, valid_dl, EPOCHS, iterations=ITERATIONS)
    b3 = benchmark_func(train_loop_hybrid, train_dl, valid_dl, EPOCHS, iterations=ITERATIONS)

    print(
        # f'Explicit loop takes: {b1[0]} +/- {b1[1]} s\n'
        # f'Itertools loop takes: {b2[0]} +/- {b2[1]} s\n'
        f'Hybrid loop takes: {b3[0]} +/- {b3[1]} s\n'
        # f'DeltaT(b1, b2) = {abs(b1[0] - b2[0])} +/- {math.sqrt(b1[1] ** 2 + b2[1] ** 2)} s\n'
        # f'DeltaT(b1, b3) = {abs(b1[0] - b3[0])} +/- {math.sqrt(b1[1] ** 2 + b3[1] ** 2)} s\n'
    )
    return


if __name__ == '__main__':
    main()


# end