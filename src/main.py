import hydra

from utils import perform_startup_checks
from models import load_model
from custom_lightning_modules import load_pl_module


@hydra.main(version_base='1.3', config_path="../conf", config_name="config")
def main(cfg):
    perform_startup_checks(cfg)
    selected_model = load_model(cfg.model)
    pl_module = load_pl_module(cfg.pl_module, selected_model)


if __name__ == "__main__":
    main()
