"""
Module for mock VAE training with wandb logging.
"""

from itertools import islice
import logging
from pathlib import Path
from typing import Any, NamedTuple
import warnings

from tqdm import tqdm
import torch
from torch.amp import GradScaler, autocast
import torch.nn as nn
import torch.optim as opt
from torch.types import Tensor
from torch.utils.data import DataLoader
from torchvision import transforms as ts
from torchvision.datasets import MNIST
from torchvision.transforms import Compose

import wandb
from wandb import Run

import spark as sp
from spark.inspect import OutputManager
from spark.processing import normalise

from BASE_train_routine import training_setup


class LatentVector(NamedTuple):
    """Latent vector container."""
    vals: Tensor
    lbls: Tensor

class TrainResults(NamedTuple):
    """
    Model trainer results. Contains the avg train and valid loss
    values and the latent space vectors with respective labels.
    NOTE:
        * the tensors in `latent_dmap` are NOT attached
          to the model Op. Graph (see OutputManager)
    """
    train_loss: list[float]
    valid_loss: list[float]
    latent_dmap: dict[int, LatentVector]


# _________________________________  BASIC VAE MODEL  _________________________________ #

class BaseEncoder(nn.Module):
    """Baseline mock encoder model."""
    def __init__(self, in_channels: int, hid_channel: int, latent_dim: int) -> None:
        super().__init__()
        self.architecture = nn.Sequential(
            nn.Conv2d(in_channels, hid_channel, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.Conv2d(hid_channel, hid_channel, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.Conv2d(hid_channel, hid_channel, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(hid_channel, hid_channel, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(hid_channel, latent_dim, kernel_size=4, stride=1),
        )
    
    def forward(self, x: Tensor) -> Tensor:
        out = self.architecture(x)
        return out

class BaseDecoder(nn.Module):
    """Baseline mock encoder model."""
    def __init__(self, latent_dim: int, hid_channel: int, out_channels: int) -> None:
        super().__init__()
        self.architecture = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, hid_channel, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hid_channel, hid_channel, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(hid_channel, hid_channel, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(hid_channel, hid_channel, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hid_channel, out_channels, kernel_size=5, stride=1),
        )
    
    def forward(self, x: Tensor) -> Tensor:
        out = self.architecture(x)
        return out

class MockVAE(nn.Module):
    """
    Baseline mock Variational AutoEncoder model for `spark` API tests and stuff.
    """
    def __init__(
        self,
        latent_dim: int,
        in_channels: int = 1,
        hid_channel: int = 32,
    ) -> None:
        super().__init__()
        self.encoder = BaseEncoder(in_channels, hid_channel, latent_dim)
        self.fc_estim_mean = nn.Linear(latent_dim, latent_dim)
        self.fc_estim_log_var = nn.Linear(latent_dim, latent_dim)
        self.decoder = BaseDecoder(latent_dim, hid_channel, in_channels)
    
    def encode_signal(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Encodes the input signal, returning mean and log variance
        estimators for the latent space prob. distribution.
        """
        out = self.encoder(x).flatten(1, -1)
        mean = self.fc_estim_mean(out)
        log_var = self.fc_estim_log_var(out)
        return mean, log_var
    
    def reparameterize(self, mean: Tensor, log_var: Tensor) -> Tensor:
        """Reparameterization Trick for latent space tractable sampling."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(mean)
        z = mean + eps * std
        return z.view(z.size(0), z.size(1), 1, 1)
    
    def forward(self, x: Tensor) -> Tensor:
        mean, log_var = self.encode_signal(x)
        z = self.reparameterize(mean, log_var)
        out = self.decoder(z)
        return out, mean, log_var


# _________________________________  LOSS FUNC for VAE  _________________________________ #

def vae_loss(pred: Tensor, target: Tensor) -> Tensor:
    """Computes VAE loss with L2 Reconstruction + KL Divergence."""
    out, mean, log_var = pred
    # proxy for Gaussian-Log LH
    l2_loss = nn.functional.mse_loss(out, target)
    # KL-div between Gauss and std Gauss
    kld_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return (l2_loss + kld_loss) / target.size(0)


# _________________________________  TRAINING ROUTINE  _________________________________ #
def safe_wandb_log(logger: Run, data: dict[str, Any], epoch: int):
    try:
        logger.log(data, step=epoch)
    except Exception as e:
        print(f"wandb.log failed (epoch={epoch}): {e}")


def train_model(
    params: dict[str, Any],
    epochs: int,
    learning_rate: float,
    train_dl: DataLoader,
    valid_dl: DataLoader,
    latent_space_hook: OutputManager,
    wandb_logger: Run,
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
    avg_train_loss, avg_valid_loss = [], []
    latent_container: dict[int, LatentVector] = {}
    tdl_len, vdl_len = map(len, (train_dl, valid_dl))

    # setup model/optimiser/scaler for memory saving
    device = params['device']
    device_type = device.type if isinstance(device, torch.device) else 'cuda'
    model = params['model'].to(device)
    loss_fn = params['loss']
    optimiser = params['optimiser'](model.parameters(), lr=learning_rate)
    scheduler = params['scheduler'](optimiser, patience=5, factor=0.5)
    scaler = GradScaler(device)

    loop = tqdm(range(epochs))
    for epoch in loop:
        
        # ---------   TRAINING   ---------
        loop.set_description('Training Model')
        model.train()
        running_batches = 0
        running_train_loss = 0.0

        for batch, (x_batch, _) in enumerate(islice(train_dl, tdl_len)):   # NOTE: ignoring labels for the moment
            loop.set_postfix({'batch': f'{batch + 1}/{tdl_len}'})

            # optimisation step
            x_batch = x_batch.to(device)

            optimiser.zero_grad()
            with autocast(device_type=device_type):
                out = model(x_batch)
                loss_val = loss_fn(out, x_batch)
            
            if check_loss_val(
                loss_val, f'Train loss NaN @ E: {epoch + 1} B: {batch + 1}',
            ):
                running_batches += 1
                running_train_loss += loss_val.item()

                scaler.scale(loss_val).backward()
                scaler.step(optimiser)
                scaler.update()
            
        # loss logging
        avg_train_loss.append(running_train_loss / max(running_batches, 1))
        

        # ---------  VALIDATION  ---------
        loop.set_description('Validating Model')
        model.eval()
        running_valid_batches = 0
        running_valid_loss = 0.0

        with torch.no_grad(), sp.forward_data_capture(model.fc_estim_mean, latent_space_hook):
            for batch, (x_batch, y_batch) in enumerate(islice(valid_dl, vdl_len)):
                loop.set_postfix({'batch': f'{batch + 1}/{vdl_len}'})

                # validation step
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)

                with autocast(device_type=device_type):
                    v_out = model(x_batch)
                    v_loss_val = loss_fn(v_out, x_batch)
                
                # add labels to hook storage
                latent_space_hook.add_labels(y_batch)
                
                if check_loss_val(
                    v_loss_val, f'Valid loss NaN @ E: {epoch + 1}',
                ):
                    running_valid_batches += 1
                    running_valid_loss += v_loss_val.item()
            
        # loss logging and lr update
        avg_v_loss_val = running_valid_loss / max(running_valid_batches, 1)
        avg_valid_loss.append(avg_v_loss_val)
        scheduler.step(avg_v_loss_val)
        # log latent space vectors (merge batches + clear storage)
        latent_container[epoch] = LatentVector(*latent_space_hook.merge())
        latent_space_hook.clear()

        # ---------  WANDB LOGGING  ---------
        safe_wandb_log(
            logger=wandb_logger,
            data={
                'train/loss': avg_train_loss[epoch],
                'train/lr': optimiser.param_groups[0]["lr"],
                'valid/loss': avg_valid_loss[epoch],
            },
            epoch=epoch,
        )
    
    results = TrainResults(avg_train_loss, avg_valid_loss, latent_container)
    return results



# _________________________________  EXECUTION  _________________________________ #

def main(
    basepath: str | Path,
    run_ID: str,
) -> None:
    # logging.basicConfig(
    #     filename='training_log.log',
    #     filemode='w',
    #     format='%(asctime)s %(levelname)s %(message)s',
    #     level=logging.INFO,
    # )
    # log = logging.getLogger(__name__)

    # -------------------  HYPERPARAMS  ------------------- #
    PREPROCESSING: Compose = ts.Compose(
        [
            ts.PILToTensor(),
            normalise,
        ]
    )
    BATCH_SIZE: int = 256
    VALID_SIZE: float = 0.3

    LATENT_DIM: int = 32
    HIDDEN_CHANNELS: int = 2 * LATENT_DIM

    EPOCHS: int = 100
    LR: int = 5e-3
    DEVICE: int = 'cuda' if torch.cuda.is_available() else 'cpu'
    LOSS_FN = vae_loss

    # -------------------  WANDB LOGGING  ------------------- #
    WANDB_PROJECT: str = 'Moch Base VAE'
    WANDB_ENTITY: str = 'edo-giancarli-sapienza-universit-di-roma'
    WANDB_RUN_ID: str = run_ID

    # -------------------  DATASET HANDLING  ------------------- #
    mnist = MNIST(root=f'{basepath}/MNIST', train=False, transform=PREPROCESSING, download=False)
    train_dl, valid_dl = sp.get_dataloaders(mnist, BATCH_SIZE, VALID_SIZE)

    # -------------------  MODEL INIT  ------------------- #
    vae = MockVAE(LATENT_DIM, hid_channel=HIDDEN_CHANNELS)

    # -------------------  TRAINING  ------------------- #
    params = training_setup(
        vae, LOSS_FN, opt.Adam, opt.lr_scheduler.ReduceLROnPlateau, DEVICE,
    )
    latent_space_hook = OutputManager()

    wandb_log: Run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        name=WANDB_RUN_ID,
        config={
            'device': DEVICE,
            'epochs': EPOCHS,
            'dataset': 'MNIST',
            'batch_size': BATCH_SIZE,
            'lr': LR,
            'architecture': 'Mock baseline VAE',
            'input_chs': 1,
            'hidden_chs': HIDDEN_CHANNELS,
            'latent_dim': LATENT_DIM,
        },
        save_code=False,
    )
    results: TrainResults = train_model(
        params=params,
        epochs=EPOCHS,
        learning_rate=LR,
        train_dl=train_dl,
        valid_dl=valid_dl,
        latent_space_hook=latent_space_hook,
        wandb_logger=wandb_log,
    )
    wandb_log.finish()

    # -------------------  SAVING STUFF  ------------------- #
    model_path = f'{basepath}/mockVAEmodel_MNIST.pt'
    if not Path(model_path).exists():
        sp.save_model(
            state_dict=vae.state_dict(),
            save_to=model_path,
            info={
                'train_loss': results.train_loss,
                'valid_loss': results.valid_loss,
            }
        )
        print('VAE mock model saved!')
    else:
        print('VAE mock model already saved!')

    latent_data_path = f'{basepath}/latentSpaceContainer_mockVAEmodel_MNIST.pt'
    latent_container = results.latent_dmap
    if not Path(latent_data_path).exists():
        torch.save(latent_container, latent_data_path)
        print('Latent space vectors saved!')
    else:
        print('Latent space vectors already saved!')
    
    return


if __name__ == '__main__':
    # - path to the data / storing directory
    # basepath: str = '/home/edoardo/Desktop/MockDataForDMs'
    basepath: str = '/mnt/d/MockDataForDMs'
    # - WANDB run ID
    run_ID: str = 'MockVAE-MNIST-mse'

    wandb.login()
    main(basepath, run_ID)


# end