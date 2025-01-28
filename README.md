# About

This project is a development environment/sandbox for Pytorch Lightning.

## Configuration

Configuration files are contained in the `conf/` folder. Contents of that folder are read by Hydra (https://hydra.cc/) to configure the application.

## Docker

The development environment is controlled via Docker. All relevant files for the Docker build process are contained in the `docker/` directory. This is primarily for local development (without any GPU-related libraries like CUDA), for help within the IDE.

Note that the VSCode `.devcontainer/devcontainer.json` file references this folder for an automated build process.

## Conda (on FASRC)

Note that on FASRC (Cannon), use the files in the `conda/` directory. The environment file (`conda/env.yaml`) has packages for interacting with GPUs (CUDA libraries) on Cannon. 

After selecting an appropriate queue, load the mamba module:
```
module load Mambaforge
```
Then create the environment with:
```
mamba env create -f conda/env.yaml
```
Then activate and use.

## Running

Hydra is configured to place output files in timestamped folders in the *current working directory*. To avoid cluttering the `src/` tree, it is best to run the entrypoint script from the root of the repository (i.e. this folder)
```
/usr/bin/python3 src/main.py [...]
```

For example, to run a RESNET18 model on the CIFAR10 dataset using an ADAM optimizer:
```
/usr/bin/python3 src/main.py \
     +model=resnet18_cifar10 \
     +dataset=cifar10 \
     +pl_module=basic_classifier \
     +optimizer=adam
```
Each of those flags dictates the config yaml file to use. For example `+model=resnet18_cifar10` says that Hydra should load the `conf/model/resnet18_cifar10.yaml` file. That config ultimately directs our code to use the CNN (a Pytorch `nn.Module`) configured in `src/models/resnet.py`.

**Restarting from a checkpoint**

If you train for a number of epochs and would like to restart from where you left off, you can run with the same call as above, but this time append `+ckpt_path=<path to model checkpoint>` to tell the Lightning module to pick up where it left off. Note that you will most likely also have to augment the number of epochs. For instance, if you choose the `last.ckpt` (i.e. the final checkpoint file) after training for 20 epochs, yet keep your config to have `trainer.max_epochs=20`, then it will not train further. To train for an additional 10 epochs (a total of 30) you can use: 

```
/usr/bin/python3 src/main.py \
     +model=resnet18_cifar10 \
     +dataset=cifar10 \
     +pl_module=basic_classifier \
     +optimizer=adam \
     +ckpt_path=/path/to/last.ckpt \
     ++trainer.max_epochs=30
```
Note the double-plus to indicate an override.

#### About the development environment

As currently written, the project contains several VSCode-specific files used to set up the development environment. 

We use the [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) extension, which can be installed directly in VSCode. After installation (globally, on your host machine), open the *folder* containing this repository.  Typically, the presence of the `.devcontainers/devcontainers.json` configuration file will trigger a prompt to "re-open the folder in a container", which you should choose. 

If not, you can click on the green "remote container" button in the lower left corner of VSCode

![](https://microsoft.github.io/vscode-remote-release/images/remote-dev-status-bar.png)

which will open the command prompt with several options; choose "Reopen in Container".

Regardless of how it's initiated, VSCode will then trigger a Docker build process (if required) and start a container. VSCode extensions (such as Python linters) can be installed *inside* the container by adding to `.devcontainers/devcontainers.json`.
