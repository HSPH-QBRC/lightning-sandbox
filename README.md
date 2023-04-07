# About

This project is a development environment/sandbox for Pytorch Lightning.

## Configuration

Configuration files are contained in the `conf/` folder. Contents of that folder are read by Hydra (https://hydra.cc/) to configure the application.

## Docker

The development environment is controlled via Docker. All relevant files for the Docker build process are contained in the `docker/` directory.

Note that the VSCode `.devcontainer/devcontainer.json` file references this folder for an automated build process.

## Running

Hydra is configured to place output files in timestamped folders in the *current working directory*. To avoid cluttering the `src/` tree, it is best to run the entrypoint script from the root of the repository (i.e. this folder)
```
/usr/bin/python3 src/main.py [...]
```

#### About the development environment

As currently written, the project contains several VSCode-specific files used to set up the development environment. 

We use the [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) extension, which can be installed directly in VSCode. After installation (globally, on your host machine), open the *folder* containing this repository.  Typically, the presence of the `.devcontainers/devcontainers.json` configuration file will trigger a prompt to "re-open the folder in a container", which you should choose. 

If not, you can click on the green "remote container" button in the lower left corner of VSCode

![](https://microsoft.github.io/vscode-remote-release/images/remote-dev-status-bar.png)

which will open the command prompt with several options; choose "Reopen in Container".

Regardless of how it's initiated, VSCode will then trigger a Docker build process (if required) and start a container. VSCode extensions (such as Python linters) can be installed *inside* the container by adding to `.devcontainers/devcontainers.json`.