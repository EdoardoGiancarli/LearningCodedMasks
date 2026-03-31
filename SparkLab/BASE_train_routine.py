"""
Module for basic training procedure.
"""

from typing import Any, Callable
import itertools as it
import warnings

from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast


def training_setup(
    model: nn.Module,
    loss: Callable,
    optimiser: Callable,
    scheduler: Callable,
    device: str,
) -> dict[str, Callable]:
    """
    Creates a container with training operations.
    NOTE:
        * the scheduler refers to the learning rate.
        * other schedulers for log train/valid loss vals,
          checkpoints, wandb are NOT accounted for as of now  
    """
    print(f'## Operating on device: {device}.')
    setup: dict[str, Any] = {
        'model': model,
        'loss': loss,
        'optimiser': optimiser,
        'scheduler': scheduler,
        'device': device,
    }
    n_params = sum(p.numel() for p in model.parameters())
    print(f'## Model parameters: {n_params}.')
    return setup


def basic_train_model(
    params: dict[str, Any],
    train_dl: DataLoader,
    epochs: int,
    learning_rate: float,
    valid_dl: DataLoader | None = None,
) -> tuple[list[float], list[float]]:
    """
    Basic train routine.
    """
    def check_loss_val(loss_val: float, msg: str) -> bool:
        """Checks the current loss value and raises a warning if is NaN."""
        if not torch.isnan(loss_val):
            return True
        warnings.warn(msg)
        return False

    # config procedure/loss container
    nbatches: int = len(train_dl)
    total_steps: int = epochs * nbatches
    avg_train_loss, avg_valid_loss = [], []

    # setup model/optimiser/scaler for memory saving
    device = params['device']
    model = params['model'].to(device)
    loss_fn = params['loss']
    optimiser = params['optimiser'](model.parameters(), lr=learning_rate)
    scheduler = params['scheduler'](optimiser, patience=5, factor=0.5)
    scaler = GradScaler(device)

    # train model
    train_iter = tqdm(
        iterable=enumerate(it.islice(it.cycle(train_dl), total_steps)),
        desc='Training Model',
    )
    running_batches = 0
    running_train_loss = 0.0

    model.train()
    for idx, (input_batch, trg_batch) in train_iter:
        # update tqdm bar to keep track of the routine
        epoch, batch = divmod(idx, nbatches)
        current_lr = optimiser.param_groups[0]['lr']
        info: dict = {
            'epoch': f'{epoch + 1}/{epochs}',
            'batch': f'{batch + 1}/{nbatches}',
            'lr': f'{current_lr:.2e}'
        }
        train_iter.set_postfix(info)

        # model optimisation
        input_batch, trg_batch = input_batch.to(device), trg_batch.to(device)

        optimiser.zero_grad()
        with autocast(device_type=device):
            out = model(input_batch)
            loss_val = loss_fn(out, trg_batch)
        
        if check_loss_val(
            loss_val, f'Train loss NaN @ E: {epoch + 1} B: {batch + 1}',
        ):
            running_batches += 1
            running_train_loss += loss_val.item()

            scaler.scale(loss_val).backward()
            scaler.step(optimiser)
            scaler.update()

        # compute avg loss val at the end of the epoch
        if (idx + 1) % nbatches == 0:
            avg_train_loss.append(running_train_loss / max(running_batches, 1))
            running_batches = 0
            running_train_loss = 0.0
        
            # --------- VALIDATION STEP ---------
            if valid_dl is not None:
                model.eval()
                running_valid_batches = 0
                running_valid_loss = 0.0

                with torch.no_grad():
                    for v_input, v_trg in valid_dl:
                        v_input, v_trg = v_input.to(device), v_trg.to(device)

                        with autocast(device_type=device):
                            v_out = model(v_input)
                            v_loss_val = loss_fn(v_out, v_trg)
                        
                        if check_loss_val(
                            v_loss_val, f'Valid loss NaN @ E: {epoch + 1}',
                        ):
                            running_valid_batches += 1
                            running_valid_loss += v_loss_val.item()
                    
                # store avg valid loss
                avg_v_loss_val = running_valid_loss / max(running_valid_batches, 1)
                avg_valid_loss.append(avg_v_loss_val)
                # update lr val through scheduler
                scheduler.step(avg_v_loss_val)

                model.train()

    return avg_train_loss, avg_valid_loss



from typing import NamedTuple
from torch.types import Tensor
from spark.inspect import forward_data_capture, OutputManager


class TrainResults(NamedTuple):
    """
    Model trainer results.
    """
    train_loss: list[float]
    valid_loss: list[float]
    latent_dmap: dict[int, Tensor]


def train_model(
    params: dict[str, Any],
    train_dl: DataLoader,
    epochs: int,
    learning_rate: float,
    latent_space_hook: OutputManager,
    valid_dl: DataLoader | None = None,
) -> TrainResults:
    """
    Basic train routine.
    """
    def check_loss_val(loss_val: float, msg: str) -> bool:
        """Checks the current loss value and raises a warning if is NaN."""
        if not torch.isnan(loss_val):
            return True
        warnings.warn(msg)
        return False

    # config procedure/loss container
    nbatches: int = len(train_dl)
    total_steps: int = epochs * nbatches
    avg_train_loss, avg_valid_loss = [], []
    latent_container: dict[int, Tensor] = {}

    # setup model/optimiser/scaler for memory saving
    device = params['device']
    model = params['model'].to(device)
    loss_fn = params['loss']
    optimiser = params['optimiser'](model.parameters(), lr=learning_rate)
    scheduler = params['scheduler'](optimiser, patience=5, factor=0.5)
    scaler = GradScaler(device)

    # train model
    train_iter = tqdm(
        iterable=enumerate(it.islice(it.cycle(train_dl), total_steps)),
        desc='Training Model',
    )
    running_batches = 0
    running_train_loss = 0.0

    model.train()
    for idx, (input_batch, trg_batch) in train_iter:
        # update tqdm bar to keep track of the routine
        epoch, batch = divmod(idx, nbatches)
        current_lr = optimiser.param_groups[0]['lr']
        info: dict = {
            'epoch': f'{epoch + 1}/{epochs}',
            'batch': f'{batch + 1}/{nbatches}',
            'lr': f'{current_lr:.2e}'
        }
        train_iter.set_postfix(info)

        # model optimisation
        input_batch, trg_batch = input_batch.to(device), trg_batch.to(device)

        optimiser.zero_grad()
        with autocast(device_type=device):
            out = model(input_batch)
            loss_val = loss_fn(out, trg_batch)
        
        if check_loss_val(
            loss_val, f'Train loss NaN @ E: {epoch + 1} B: {batch + 1}',
        ):
            running_batches += 1
            running_train_loss += loss_val.item()

            scaler.scale(loss_val).backward()
            scaler.step(optimiser)
            scaler.update()

        # compute avg loss val at the end of the epoch
        if (idx + 1) % nbatches == 0:
            avg_train_loss.append(running_train_loss / max(running_batches, 1))
            running_batches = 0
            running_train_loss = 0.0
        
            # ------------- validation step -------------
            if valid_dl is not None:
                model.eval()
                running_valid_batches = 0
                running_valid_loss = 0.0

                with torch.no_grad(), forward_data_capture(model.encoder, latent_space_hook):
                    for v_input, v_trg in valid_dl:
                        v_input, v_trg = v_input.to(device), v_trg.to(device)

                        with autocast(device_type=device):
                            v_out = model(v_input)
                            v_loss_val = loss_fn(v_out, v_trg)
                        
                        if check_loss_val(
                            v_loss_val, f'Valid loss NaN @ E: {epoch + 1}',
                        ):
                            running_valid_batches += 1
                            running_valid_loss += v_loss_val.item()
                    
                # store avg valid loss
                avg_v_loss_val = running_valid_loss / max(running_valid_batches, 1)
                avg_valid_loss.append(avg_v_loss_val)
                # update lr val through scheduler
                scheduler.step(avg_v_loss_val)
                # merge latent space vectors for current epoch and clear storage
                latent_container[epoch] = latent_space_hook.merge()
                latent_space_hook.clear()

                model.train()

    results = TrainResults(avg_train_loss, avg_valid_loss, latent_container)

    return results


# end