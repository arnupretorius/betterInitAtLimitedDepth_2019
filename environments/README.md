# Research environment

For now we might be using different research enviroments, so this directory serves as a location for each of these until we decide on a coherent solution.

## Conda:

How to set up the Anaconda environment:

1. Install [Anaconda](https://www.anaconda.com/download)
2. Navigate your terminal to the root directory of this repo
3. Create the environment with the following command: `conda env create -f env.yml`
4. Activate the environment with the following command: `source activate torch`

## Docker:

The `Dockerfile` is based on a minimal [PyTorch image](https://hub.docker.com/r/reloff/pytorch-base/) (torch==0.4.1; python==3.6; CUDA==9.0). Dependencies are installed from the accompanying `requirements.txt`. Simply add/change/remove dependecies and re-build the Docker image (currently located at https://hub.docker.com/r/reloff/noisy-relu-shift/).

Run experiment notebooks in the Docker research environment:

1. Install [Docker](https://docs.docker.com/install/) (recommend following the [linux post-install step](https://docs.docker.com/install/linux/linux-postinstall/) to manage Docker as a non-root user)
2. Install [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) (version 2.0) for NVIDIA GPU access in docker containers
3. Pull research image from [Docker Hub](https://hub.docker.com/r/reloff/noisy-relu-shift/): `docker pull reloff/noisy-relu-shift`\
    Or build research image from the Dockerfile: `docker build -t noisy-relu-shift ./docker`
4. Run experiment notebooks (from the repo root): `./run_notebooks.sh`\
    Or run with options (see `./run_notebooks.sh --help` for more information):
    ```bash
    ./run_notebooks.sh \
        --port=8888 \
        --image=reloff/noisy-relu-shift
        --name=noisy-relu-shift \
        --password=1234567890
    ```
