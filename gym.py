import click
import yaml
import torch
import numpy as np
from typing import Any, Dict, List, Optional

from data.nifti_loader import MedicalImageDatasetSplitter, MonaiDatasetCreator, MonaiDataLoaderManager
from config.config_loader import load_config
from models.cnn_backbones import Small3DCNN
from training.trainer import train_model, test_model

# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------
def set_seed(seed: int = 42) -> None:
    """
    Set global random seed for reproducibility across numpy and torch.

    :param seed: Seed value to use.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def apply_overrides(cfg: Dict[str, Any], overrides: List[str]) -> None:
    """
    Apply list of "key.path=value" overrides to the nested config dict.

    :param cfg: Configuration dictionary to update.
    :param overrides: List of override strings.
    """
    for ov in overrides:
        keypath, raw_val = ov.split('=', 1)
        value = yaml.safe_load(raw_val)
        keys = keypath.split('.')
        subcfg = cfg
        for k in keys[:-1]:
            if k not in subcfg or not isinstance(subcfg[k], dict):
                raise click.BadParameter(f"Invalid override key: '{keypath}'")
            subcfg = subcfg[k]
        subcfg[keys[-1]] = value

# -----------------------------------------------------------------------------
# CLI Definition
# -----------------------------------------------------------------------------
@click.command()
@click.option(
    "--config-path", "-c",
    default="config/base_config.yaml",
    type=click.Path(exists=True),
    show_default=True,
    help="Path to the YAML configuration file."
)
# First-class overrides
@click.option("--batch-size",          type=int,                      help="Override config['data']['batch_size']")
@click.option("--perform-slicing/--no-slicing", default=None,                help="Enable or disable slicing")
@click.option("--image-size", nargs=3, type=int,                              help="Three ints for config['data']['image_size']")
@click.option("--epochs",              type=int,                      help="Override config['training']['epochs']")
# Generic override for any field
@click.option(
    "--override", "-o",
    multiple=True,
    help="Generic override in key.path=value format; repeatable."
)
def main(
    config_path: str,
    batch_size: Optional[int],
    perform_slicing: Optional[bool],
    image_size: Optional[List[int]],
    epochs: Optional[int],
    override: List[str]
) -> None:
    """
    Entrypoint: load config, apply overrides, prepare data, train, and test.
    """
    # Reproducibility
    set_seed()

    # Load base config
    cfg: Dict[str, Any] = load_config(config_path)

    # Apply first-class overrides
    if batch_size is not None:
        cfg['data']['batch_size'] = batch_size
    if perform_slicing is not None:
        cfg['data']['perform_slicing'] = perform_slicing
    if image_size is not None:
        cfg['data']['image_size'] = list(image_size)
    if epochs is not None:
        cfg['training']['epochs'] = epochs

    # Apply generic overrides
    if override:
        apply_overrides(cfg, list(override))

    # Data pipeline
    splitter = MedicalImageDatasetSplitter(cfg)
    creator = MonaiDatasetCreator(splitter)
    manager = MonaiDataLoaderManager(creator, cfg)
    loaders = manager.get_dataloaders()

    # Model setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Small3DCNN(num_classes=splitter.get_num_classes()).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg['training'].get('lr', 1e-3)
    )

    # Training
    trained_model, history = train_model(
        model=model,
        train_loader=loaders['train'],
        val_loader=loaders['val'],
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=cfg['training']['epochs'],
        device=device
    )
    click.echo("\nTraining Completed:")
    click.echo(f"  Train Loss: {history['train_loss']}")
    click.echo(f"  Train Acc:  {history['train_accuracy']}")
    click.echo(f"  Val Loss:   {history['val_loss']}")
    click.echo(f"  Val Acc:    {history['val_accuracy']}")

    # Testing
    test_loss, test_acc, _, _ = test_model(
        trained_model, loaders['test'], criterion
    )
    click.echo(f"\nTest -> Loss: {test_loss:.4f}, Acc: {test_acc:.4f}")


if __name__ == "__main__":
    main()