import os

import hydra


@hydra.main(version_base='1.3', config_path="../conf", config_name="config")
def main(cfg):
    print(f'Output dir: {os.getcwd()}')


if __name__ == "__main__":
    main()
