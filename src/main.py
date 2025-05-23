import hydra
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, \
    ModelCheckpoint

from utils.general import perform_startup_checks
from models import load_model
from custom_lightning_modules import load_pl_module
from data_modules import load_dataset

torch.set_float32_matmul_precision('medium')

@hydra.main(version_base='1.3', config_path="../conf", config_name="config")
def main(cfg):
    perform_startup_checks(cfg)
    selected_model = load_model(cfg.model)
    pl_module = load_pl_module(cfg, selected_model)
    datamodule = load_dataset(cfg.dataset)
    trainer = Trainer(accelerator='auto',
                      devices='auto',
                      max_epochs=cfg.trainer.max_epochs,
                      log_every_n_steps=5,
                      accumulate_grad_batches=cfg.trainer.grad_acc,
                      callbacks=[
                          LearningRateMonitor(logging_interval="step"),
                          ModelCheckpoint(
                              filename='{epoch}-{val_loss:.2f}-{other_metric:.2f}',
                              save_last=True,
                              monitor='val_loss',
                              mode='min'
                          )
                        ])
    
    # to restart from a saved checkpoint:
    if 'ckpt_path' in cfg:
        trainer.fit(model=pl_module, datamodule=datamodule, ckpt_path=cfg.ckpt_path)
    else:
        trainer.fit(model=pl_module, datamodule=datamodule)


if __name__ == "__main__":
    main()
