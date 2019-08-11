# The effect of initialisation on learning dynamics, expressivity and generalisation in deep noisy rectifier networks

**Team**: Arnu Pretorius, Steve James, Benji Van Niekerk, Ryan Eloff, Elan Van Biljon, Matthew Reynard, Herman Kamper, Steve Kroon, Herman Engelbrecht, Benji Rosman

### Basic idea:

We investigate how learning, expressivity and generalisation of noisy rectifier neural networks are affected by their initialisation. This work aims to build on the publication below.

[1] Pretorius, A., Van Biljon, E., Kroon, S., Kamper, H. Critical initialisation for deep signal propagation in noisy rectifier neural networks. In advances in neural information processing systems, 2018.


## Project overview

In [1] we derived optimal initialisation schemes for random noisy neural networks. However, the theory used in the paper is only valid at initialisation. Therefore, in this proposed work we attempt to expand the investigation to see what happens during training. Specifically, we seek to answer some of the following research questions:

* What effect does initialisation have on the learning dynamics, expressivity and generalisation of fully connected feedforward rectifier neural networks that use noise regularisation (e.g. dropout)?
* Does the optimal initialisation provide any benefit after the initial forward pass of the network, i.e. during training when the network is no longer random?
* Given that we restrict the depth of the network to ensure that ReLU type (unbounded) activations do not over/underflow before reaching the output layer, are there initialisations off the critical point that might provide better performance? For example, lower variance inits that might induce implicit biases towards more parsimonious models.

## Steps for running experiments (GPU required):

Below are the instructions to run experiments on a machine

### Docker experiment instructions

(See below for instructions on running experiments in Conda environment)

#### Step 1. Install [Docker](https://docs.docker.com/engine/installation/) and [nvidia-docker](https://github.com/NVIDIA/nvidia-docker), if not installed already.

#### Step 2. Obtain the research environment image from [Docker Hub](https://hub.docker.com/r/reloff/noisy-relu-shift/).

```bash
docker pull reloff/noisy-relu-shift
```
(Or, alternatively, build the image in [environments/docker](https://github.com/arnupretorius/noisy_relu_shift/tree/master/environments/docker))

#### Step 3. Clone the research code repository.
```bash
git clone https://github.com/arnupretorius/noisy_relu_shift.git
```

#### Step 4. Generate experimental results (**Warning: this may take several hours to run.**)

Run experiments in a Docker container with `./run_experiments --docker [options]` (use `--help` flag for more information)

*Usage:*

```bash
./run_experiments.sh \
    --docker \
    --hp=<hyperparams_file> \
    --exp=<experiments_file> \
    --act=<act> \
    --dataset=<dataset> \
    --epochs=<epochs>
```

where `<hyperparams_file>` is the hyperparams specification file, `<experiments_file>` is the experiments specification file, `<act>` is the activation function, `<dataset>` is the name of the dataset and `<epochs>` is the number of training epochs. The `--docker` flag specifies that experiments will be run in a Docker container with the research image pulled/built in Step 2. For now we set these to `hyperparams_30.txt jobs_per_pc/no_gauss/experiment_1.txt relu cifar10 500`, i.e. run the following:

```bash
./run_experiments.sh --docker --hp=hyperparams_30.txt --exp=jobs_per_pc/no_gauss/experiment_1.txt --act=relu --dataset=cifar10 --epochs=500
```

### Conda experiment instructions

#### Step 1. Install [Anaconda](https://www.anaconda.com/download)

#### Step 2. Create the environment with the following command: `conda env create -f environments/conda/env.yml`

#### Step 3. Activate the environment with the following command: `source activate torch`

#### Step 4. Generate experimental results (**Warning: this may take several hours to run.**)

Run experiments in current Conda environment with `./run_experiments [options]` (use `--help` flag for more information)

*Usage:*

```bash
./run_experiments.sh \
    --hp=<hyperparams_file> \
    --exp=<experiments_file> \
    --act=<act> \
    --dataset=<dataset> \
    --epochs=<epochs>
```

where `<hyperparams_file>` is the hyperparams specification file, `<experiments_file>` is the experiments specification file, `<act>` is the activation function, `<dataset>` is the name of the dataset and `<epochs>` is the number of training epochs. For now we set these to `hyperparams_30.txt jobs_per_pc/no_gauss/experiment_1.txt relu cifar10 500`, i.e. run the following:

```bash
./run_experiments.sh --hp=hyperparams_30.txt --exp=jobs_per_pc/no_gauss/experiment_1.txt --act=relu --dataset=cifar10 --epochs=500
```
