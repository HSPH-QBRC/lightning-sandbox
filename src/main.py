import hydra
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor

from utils import perform_startup_checks
from models import load_model
from custom_lightning_modules import load_pl_module
from data_modules import load_dataset


@hydra.main(version_base='1.3', config_path="../conf", config_name="config")
def main(cfg):
    perform_startup_checks(cfg)
    selected_model = load_model(cfg.model)
    pl_module = load_pl_module(cfg, selected_model)
    datamodule = load_dataset(cfg.dataset)
    trainer = Trainer(accelerator='auto',
                      devices='auto',
                      max_epochs=cfg.trainer.max_epochs,
                      callbacks=[
                          LearningRateMonitor(logging_interval="step")])
    trainer.fit(model=pl_module, datamodule=datamodule)


if __name__ == "__main__":
    main()
