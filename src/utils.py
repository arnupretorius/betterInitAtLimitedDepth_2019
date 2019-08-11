# https://github.com/hunkim/PyTorchZeroToAll/blob/master/09_2_softmax_mnist.py

import os, torch
import numpy as np
import torch.optim as optim

from tqdm import tqdm
from src.net import Net
from torchvision import transforms
from src.cca_core import get_cca_similarity
from src.data_loader import load_data, get_data_dimensions

from tqdm import tqdm

def get_settings():
    # set device and data type
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    dtype = torch.float32
    dtype_y = torch.long
    return device, dtype, dtype_y

def train_epoch(model, train_loader, test_loader, criterion, optimiser, device, dtype, dtype_y, name, epoch, model_dir, noise_type, noise_level):
    model.train()
    for data, target in train_loader:
        data, target = data.to(device, dtype), target.to(device, dtype_y)

        optimiser.zero_grad()
        output = model(data)
        train_loss_obj = criterion(output, target)

        if torch.isnan(train_loss_obj):
            save_model(model, name, epoch, noise_type, noise_level, nan=True, model_dir=model_dir)
            return True

        train_loss_obj.backward()
        optimiser.step()

    return False

def evaluate_model(model, train_loader, test_loader, criterion, device, dtype, dtype_y):
    train_loss = 0.0
    train_acc = 0.0
    test_loss = 0.0
    test_acc = 0.0

    model.eval()
    with torch.no_grad():
        # check train progress
        for data, target in train_loader:
            data, target = data.to(device, dtype), target.to(device, dtype_y)

            output = model(data)
            train_loss += criterion(output, target)

            predictions = torch.argmax(output, dim=-1)
            train_acc += torch.sum(predictions == target, dtype=dtype)

        train_divisor = len(train_loader.dataset)
        train_loss /= train_divisor
        train_acc /= train_divisor

        # check test progress
        for data, target in test_loader:
            data, target = data.to(device, dtype), target.to(device, dtype_y)

            output = model(data)
            test_loss += criterion(output, target)

            predictions = torch.argmax(output, dim=-1)
            test_acc += torch.sum(predictions == target, dtype=dtype)

        test_divisor = len(test_loader.dataset)
        test_loss /= test_divisor
        test_acc /= test_divisor

    train_loss = train_loss.data.item()
    train_acc = train_acc.data.item()
    test_loss = test_loss.data.item()
    test_acc = test_acc.data.item()

    return train_loss, train_acc, test_loss, test_acc

def save_training_stats(name, results_dir, train_loss, train_acc, test_loss, test_acc):
    train_loss_file_name = "{}_train_loss.npy".format(name)
    train_loss_file_path = os.path.join(results_dir, train_loss_file_name)

    train_acc_file_name = "{}_train_accuracy.npy".format(name)
    train_acc_file_path = os.path.join(results_dir, train_acc_file_name)

    test_loss_file_name = "{}_test_loss.npy".format(name)
    test_loss_file_path = os.path.join(results_dir, test_loss_file_name)

    test_acc_file_name = "{}_test_accuracy.npy".format(name)
    test_acc_file_path = os.path.join(results_dir, test_acc_file_name)

    if os.path.exists(train_loss_file_path):
        train_losses = np.load(train_loss_file_path)
    else:
        train_losses = np.array([])

    if os.path.exists(train_acc_file_path):
        train_accs = np.load(train_acc_file_path)
    else:
        train_accs = np.array([])

    if os.path.exists(test_loss_file_path):
        test_losses = np.load(test_loss_file_path)
    else:
        test_losses = np.array([])

    if os.path.exists(test_acc_file_path):
        test_accs = np.load(test_acc_file_path)
    else:
        test_accs = np.array([])

    train_losses = np.append(train_losses, [train_loss])
    train_accs = np.append(train_accs, [train_acc])
    test_losses = np.append(test_losses, [test_loss])
    test_accs = np.append(test_accs, [test_acc])

    np.save(train_loss_file_path, train_losses)
    np.save(train_acc_file_path, train_accs)
    np.save(test_loss_file_path, test_losses)
    np.save(test_acc_file_path, test_accs)

def train(model, train_loader, test_loader, criterion, optimiser, epochs, noise_type, noise_level, save=False, model_dir='../results', name=None, results_dir="../plotting/training", start_epoch=0):
    if name is None:
        raise ValueError("Must give the model a name in the `train` function.")

    device, dtype, dtype_y = get_settings()

    # get starting model stats and set initial train accuracy threshold
    train_loss, train_acc, test_loss, test_acc = evaluate_model(model, train_loader, test_loader, criterion, device, dtype, dtype_y)
    save_training_stats(name, results_dir, train_loss, train_acc, test_loss, test_acc)

    desc = "Epoch: {:3d}, Train loss: {:1.4E}, Train accuracy: {:1.3f}, Test loss: {:1.4E}, Test accuracy: {:1.3f}".format(
        start_epoch, train_loss, train_acc, test_loss, test_acc
    )

    accuracy_thresholds = np.linspace(0.1, 1.0, 10)
    accuracy_thresholds = np.sort(np.append(accuracy_thresholds, [0.95, 0.98, 0.99]))
    # there is actually a bug here and on line 158 but since experiments have been running like this, we have opted to leave it in, it's not a big deal anyway
    max_threshold_index = np.sum(train_acc >= accuracy_thresholds) # - 1

    checkpoint_epochs = np.linspace(0, epochs, 4, endpoint=False, dtype=int)

    with tqdm(total=(epochs - start_epoch)) as progress_bar:
        progress_bar.set_description(desc=desc)

        for epoch in range(start_epoch+1, epochs+1):
            nan = train_epoch(model, train_loader, test_loader, criterion, optimiser, device, dtype, dtype_y, name, epoch, model_dir, noise_type, noise_level)

            if nan:
                return model

            train_loss, train_acc, test_loss, test_acc = evaluate_model(model, train_loader, test_loader, criterion, device, dtype, dtype_y)
            save_training_stats(name, results_dir, train_loss, train_acc, test_loss, test_acc)

            threshold_index = np.sum(train_acc >= accuracy_thresholds) # - 1

            if threshold_index > max_threshold_index:
                max_threshold_index = threshold_index
                save_model(model, name, epoch, noise_type, noise_level, model_dir=model_dir)
            elif epoch in checkpoint_epochs:
                save_model(model, name, epoch, noise_type, noise_level, model_dir=model_dir)

            desc = "Epoch: {:3d}, Train loss: {:1.4E}, Train accuracy: {:1.3f}, Test loss: {:1.4E}, Test accuracy: {:1.3f}".format(
                epoch, train_loss, train_acc, test_loss, test_acc
            )
            progress_bar.set_description(desc=desc)

            if train_acc == 1:
                progress_bar.update(epochs - epoch)
                break

            progress_bar.update(1)

    save_model(model, name, epoch, noise_type, noise_level, model_dir=model_dir)
    return model

def get_acc(model, data_loader):
    device, dtype, dtype_y = get_settings()
    accuracy = 0.0

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device, dtype), target.to(device, dtype_y)
            output = model(data)
            predictions = torch.argmax(output, dim=-1)
            accuracy += torch.sum(predictions == target, dtype=dtype)

    return accuracy / len(data_loader.dataset)


def test(model, test_loader, criterion, quiet=False):
    device, dtype, dtype_y = get_settings()
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device, dtype), target.to(device, dtype_y)
            output = model(data)
            test_loss += criterion(output, target).data.item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()

    test_loss /= len(test_loader.dataset)
    if not quiet:
        print('\nTest: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
    # return test loss and test accuracy
    return test_loss, float(correct) / float(len(test_loader.dataset))

def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

def get_number_of_experiments(experiments_file, hyperparams_file):
    n_hyperparams = file_len(hyperparams_file)
    n_experiments = file_len(experiments_file)
    return n_hyperparams*n_experiments

# def hp_dict():
#     '''Return the dictionary of the possible hyperparameters'''
#     return {
#         "index": (),
#         "batch": (32, 64, 128, 256, 512, 1024),
#         "depth": (2, 4, 6, 8, 10, 15, 20, 25, 30, 40, 50, 75, 100), # num of hidden layers
#         "width": (400, 450, 500, 600, 750, 900),  # size of each hidden layers
#         "seed": (),
#         "learning_rate": (10**-1, 10**-2, 10**-3, 10**-4, 10**-5, 10**-6, 10**-7),
#         "momentum": (0, 0.9),
#         "optimiser": ("SGD", "Adam", "RMSprop")
#     }

def fixed_point(f, guess, sigma=1, mu2=1, epsilon=10**(-8), n=1000):
    itr=0
    fp = 0
    test=f(guess, sigma, mu2)
    if (abs(test-guess)<epsilon):
        fp = guess
    while ((n>itr) and (abs(test-guess)>=epsilon)):
        itr+=1
        guess = test
        test = f(test, sigma, mu2)
        if ((abs(test-guess))<epsilon):
            fp = guess
    return fp

def c_map(x, sigma=1, mu2=1):
    return (sigma/(2*np.pi))*(x*np.arcsin(x) + np.sqrt(1-x**2)) + (sigma/4)*x + 1 - (sigma/2)*mu2

def c_map_slope(x, sigma):
    return (sigma/(2*np.pi))*np.arcsin(x) + sigma/4

def depth_scale(xi):
    return -1/(np.log(xi))

def get_trainable_depth(noise_type, noise_level):
    noise_type = noise_type.lower()
    noise_level = float(noise_level)

    mu2 = mu_2(noise_type, noise_level)
    critical = get_critical_init(noise_type, noise_level)
    fpoint = fixed_point(c_map, 1, critical, mu2)
    fp_slope = c_map_slope(fpoint, critical)
    depth = 6 * depth_scale(fp_slope)

    # put a cap on the depth of networks we want to train
    if np.isinf(depth):
        depth = 20
        # depth = np.max(hp_dict()["depth"])

    return int(np.floor(depth))

def hp_dict(noise_type, noise_level):
    noise_level = float(noise_level) if noise_level is not None else 0
    noise_type = noise_type if noise_type is not None else "none"

    '''Return the dictionary of the possible hyperparameters'''
    # never sample a depth greater than the max trainable depth for this noise configuration
    if noise_type != "none":
        max_depth = get_trainable_depth(noise_type, noise_level)
        depth = list(map(int, np.linspace(2, max_depth, num=(max_depth-1), dtype=int)))
    else:
        depth = (2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20)

    return {
        "index": (),
        "batch": (32, 64, 128, 256),
        "depth": depth, # num of hidden layers
        "width": (400, 600, 800), # size of each hidden layers
        "seed": (),
        "learning_rate": (1e-3, 1e-4, 1e-5, 1e-6),
        "momentum": (0, 0.5, 0.9),
        "optimiser": ("SGD", "Adam", "RMSprop")
    }
# def hp_dict():
#     '''Return the dictionary of the possible hyperparameters'''
#     return {
#         "index": (),
#         "batch": (32, 64, 128, 256),
#         "depth": np.linspace(2, 20, num=(20-1), dtype=int), # num of hidden layers
#         "width": (400, 600, 800), # size of each hidden layers
#         "seed": (),
#         "learning_rate": (1e-3, 1e-4, 1e-5, 1e-6),
#         "momentum": (0, 0.5, 0.9),
#         "optimiser": ("SGD", "Adam", "RMSprop")
#     }

def exp_dict():
    return {
        "none": [0],
        "dropout": [0.9, 0.5, 0.7],
        "gauss": [0.5, 1, 2]
    }

def get_hyperparameters(indices, noise_type, noise_level):
    hps = hp_dict(noise_type, noise_level)
    hyperparams = []
    for index, (key, value) in zip(indices, hps.items()):
        if key =="seed" or key == "index":
            hyperparams.append(index) # not the most elegant solution
        else:
            try:
                hyperparams.append(value[index])
            except:
                print(indices)
                print(index)
                print(value)
    return hyperparams

def get_experiment(indices, n_layers):
    exps = exp_dict()
    noise_type = list(exps.keys())[indices[0]]
    noise_level = exps[noise_type][indices[1]]
    init_val = get_initialisations(noise_type, noise_level, n_layers)[indices[2]]
    return noise_type, noise_level, init_val

def get_noise(indices):
    exps = exp_dict()
    noise_type = list(exps.keys())[indices[0]]
    noise_level = exps[noise_type][indices[1]]
    return noise_type, noise_level

def sample_hp(noise_type, noise_level):
    '''Given a dictionary of possible hyperparameters, return a dict of logical hyperparams to use'''
    hp = hp_dict(noise_type, noise_level)

    sampled_hp = {}

    sampled_hp["optimiser"] = np.random.randint(len(hp["optimiser"]))
    sampled_hp["learning_rate"] = np.random.randint(len(hp["learning_rate"]))
    sampled_hp["momentum"] = np.random.randint(len(hp["momentum"]))
    sampled_hp["batch"] = np.random.randint(len(hp["batch"]))
    sampled_hp["width"] = np.random.randint(len(hp["width"]))
    sampled_hp["seed"] = np.random.randint(100)

    # # never sample a depth greater than the max trainable depth for this noise configuration
    # max_depth = get_trainable_depth(noise_type, noise_level)
    # sampled_hp["depth"] = np.random.randint(len(hp["depth"][hp["depth"] <= max_depth]))
    sampled_hp["depth"] = np.random.randint(len(hp["depth"]))

    return sampled_hp

def lcm(array):
    return np.lcm.reduce(array)

def shuffle_int(num):
    array = list(range(num))
    np.random.shuffle(array)
    return array

def shuffle(array):
    np.random.shuffle(array)
    return array

def uniform_random_hps(num, noises):
    _lcms = []
    for (noise_type, noise_level) in noises:
        full_hp_set = hp_dict(noise_type, noise_level)
        lengths = [len(values) if len(values) > 0 else 1 for key, values in full_hp_set.items()]

        # lowest common multiple
        _lcms.append(lcm(lengths))

    # check if number of hyperparams is valid (must be a multiple of all the _lcms):
    for _lcm in _lcms:
        if num / _lcm != num // _lcm:
            print("lowest valid value for num is:", lcm(_lcms))
            raise ValueError("num is not valid, need a multiple of {}".format(_lcms))

    _lcm = 12
    random_hps = {}

    for key, value in full_hp_set.items():
        if len(value) == 0:
            random_hps[key] = [np.random.randint(100) for _ in range(num)]
        else:
            num_lists = range(num // _lcm)

            if key == "depth":
                random_hps["depth"] = {}
                for noise_type, noise_level in noises:
                    full_hp_set = hp_dict(noise_type, noise_level)
                    length = len(full_hp_set[key])
                    num_repetitions = range(_lcm // length)
                    rh_key = "{} {}".format(noise_type, noise_level)
                    random_hps["depth"][rh_key] = list(np.array([
                        [shuffle_int(length) for _i in num_repetitions]
                        for _j in num_lists
                    ]).reshape((-1,)))
            else:
                length = len(full_hp_set[key])
                num_repetitions = range(_lcm // length)
                random_hps[key] = list(np.array([
                    [shuffle_int(length) for _i in num_repetitions]
                    for _j in num_lists
                ]).reshape((-1,)))

            # _lcm = _lcms[noise_index]
            # num_lists = int(num / _lcm)
            # num_repetitions = int(_lcm / length)

            # random_hps[rh_key][key] = [[list(range(length)) for _i in range(num_repetitions)] for _j in range(num_lists)]
            # random_hps[rh_key][key] = list(np.array(random_hps[rh_key][key]).reshape((-1,)))
            # for i in range(num_lists * num_repetitions):
            #     section = random_hps[rh_key][key][length*i:length*(i+1)]
            #     np.random.shuffle(section)
            #     random_hps[rh_key][key][length*i:length*(i+1)] = section

    return random_hps

def sample_optimiser(op, params, learning_rate, momentum):
    '''Return the pytorch optimiser, given the sampled hyperparameters'''
    if op == "SGD":
        return optim.SGD(params, lr=learning_rate, momentum=momentum)
    elif op == "Adam":
        return optim.Adam(params, lr=learning_rate)
    elif op == "RMSprop":
        return optim.RMSprop(params, lr=learning_rate, momentum=momentum)
    else:
        print("Invalid optimiser")
        return None


def create_exp_file(split=1):
    masterEXP = np.array([], dtype=np.int32)
    exp = exp_dict()
    num_inits = len(get_initialisations('none', 0, 100))
    noise_types = exp.keys()
    total = 0
    for i, noise_type in enumerate(noise_types):
        for j, noise_level in enumerate(exp[noise_type]):
            for k in range(num_inits):
                masterEXP =  np.append(masterEXP, [i, j, k])
                total += 1

    masterEXP = np.reshape(masterEXP, (-1, 3))
    np.savetxt("./experiments.txt", masterEXP, fmt="%d", delimiter=" ")

    print(total)
    print(int(total/split))
    if split>1:
        os.makedirs('./jobs', exist_ok=True)
        t = int(total/split) # total hp in each file
        masterEXP = list(masterEXP)
        while len(masterEXP) != 0:
            for i in range(split):
                path = "./jobs/experiment_{}.txt".format(i+1)
                with open(path, "a") as job:
                    job.write(' '.join(map(str, masterEXP.pop(0))) + '\n')
                if len(masterEXP) == 0:
                    break

def create_hp_file(noises, total=72, split=1, file_path="../experiments/commands/all.txt"):
    '''Create the master hyperparameters file with indices'''
    # masterHP = np.array([], dtype=np.int32)

    np.random.seed(0)
    # for i in range(total):
    hp = uniform_random_hps(total, noises)

    # noise_translator = exp_dict()
    noise_translator = {"none": 0, 0: 0, "dropout": 1, 0.5: 1, 0.7: 2, 0.9: 0}

    lines = []

    for i in range(total):
        for (noise_type, noise_level) in noises:
            noise_key = "{} {}".format(noise_type, noise_level)

            for j in range(len(get_initialisations(noise_type, noise_level, 10))):
                for dataset in ("mnist", "cifar10"):
                    line = "python experiment.py {}_{}_{}_{}_{}_{}_{}_{} {}_{}_{} {}\n".format(
                        i+31,
                        hp["batch"][i],
                        hp["depth"][noise_key][i],
                        hp["width"][i],
                        hp["seed"][i],
                        hp["learning_rate"][i],
                        hp["momentum"][i],
                        hp["optimiser"][i],
                        noise_translator[noise_type],
                        noise_translator[noise_level],
                        j,
                        dataset
                    )
                    lines.append(line)

    # shuffle sections within the files
    section_size = 88 * 4
    num_sections = int(np.ceil(len(lines) / section_size))
    for section_index in range(num_sections):
        start = section_index * section_size
        end = (section_index + 1) * section_size
        lines[start:end] = shuffle(lines[start:end])

    with open(file_path, "w") as f:
        f.write("")
    # with open("new_commands.txt", "w") as f:
    #     f.write("")

    with open(file_path, "a") as f:
    # with open("new_commands.txt", "a") as f:
        for line in lines:
            f.write(line)

    # masterHP = np.reshape(masterHP, (-1, 10))
    # # masterHP = np.reshape(masterHP, (-1, 8))
    # np.savetxt("./hyperparams_new.txt", masterHP, fmt="%d", delimiter=" ")

    # if split > 1:
    #     os.makedirs('./hyperparameters', exist_ok=True)
    #     t = int(total/split) # total hp in each file

    #     for i in range(split):
    #         path = "./hyperparameters/hyperparams{}.txt".format(i+1)
    #         np.savetxt(path, masterHP[i*t:i*t+t], fmt="%d", delimiter=" ")


def read_hp_file(noise_type, noise_level, create_readme=True):
    '''Test: to read the info in the master hyperparameters file'''
    x = np.loadtxt("./hyperparams.txt", dtype=np.int32, delimiter=" ")

    if create_readme:
        hp = hp_dict(noise_type, noise_level)
        a = np.array([])
        a = np.append(a, "These are the current hyperparameters of each experiment in human readable format\n")
        a = np.append(a, "n: number, b: batch size, d: depth, w: width, s: seed, lr: learning rate, m: momentum, op: optimiser, nt: noise type, nl: noise level, i: init value e: epoch\n")
        a = np.append(a, "File naming format: \"n_b_d_w_s_lr_m_op_nt_nl_i_e.pth\"\n\n")
        for i in range(len(x)):
            a = np.append(a, "{}: b:{}, d:{}, w:{}, s:{}, lr:{}, m:{}, op:{}".format(x[i][0],
                                                                        hp["batch"][x[i][1]],
                                                                        hp["depth"][x[i][2]],
                                                                        hp["width"][x[i][3]],
                                                                        x[i][4],
                                                                        hp["learning_rate"][x[i][5]],
                                                                        hp["momentum"][x[i][6]],
                                                                        hp["optimiser"][x[i][7]]))
        path = "./HP_README.txt"
        np.savetxt(path, a, fmt="%s")

    return x

def hyperparam_indices_from_index(index):
    index = int(index)
    base_path = os.path.dirname(os.path.abspath(__file__))
    hyperparams_file_path = os.path.join(base_path, "../experiments/hyperparams.txt")
    with open(hyperparams_file_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        indices = list(map(int, line.strip().split(" ")))
        if index == indices[0]:
            return indices

    raise ValueError("No set of hyper-params found to match the given index.")

def variance_prop_depth(noise_type, sigma, noise_level=None, q_0=1):
    _mu = mu_2(noise_type, noise_level)
    growth_rate = sigma * _mu / 2

    if isinstance(growth_rate, float):
        if growth_rate < 1:
            value = np.finfo("float32").tiny
        else:
            value = np.finfo("float32").max

        return (np.log10(value) - np.log10(q_0))/np.log10(growth_rate)

    elif isinstance(growth_rate, np.ndarray):
        explode_value = np.finfo("float32").max
        shrink_value = np.finfo("float32").tiny

        ret_val = np.empty(growth_rate.shape)

        exploding_ps = growth_rate >= 1
        ret_val[exploding_ps] = (np.log10(explode_value) - np.log10(q_0))/np.log10(growth_rate[exploding_ps])

        shrinking_ps = growth_rate < 1
        ret_val[shrinking_ps] = (np.log10(shrink_value) - np.log10(q_0))/np.log10(growth_rate[shrinking_ps])

        return ret_val

    else:
        raise ValueError("growth rate of type {} not supported, check that you have valid values for noise_level and sigma".format(type(growth_rate)))

def mu_2(noise_type, noise_level):
    if isinstance(noise_type, str):
        dist = noise_type.lower()
        if "none" in dist:
            return 1
        elif "gauss" in dist:
            return noise_level + 1
        elif "drop" in dist or "bern" in dist:
            return 1 / noise_level
        else:
            raise ValueError("No mu_2 calculations have been added for the '{}' noise type".format(noise_type))
    else:
        raise TypeError("noise_type must be a string")

def get_critical_init(noise_type, noise_level):
    return 2 / mu_2(noise_type, noise_level)

def get_initialisations(noise_type, noise_level, n_layers, give_boundaries=False, old_experiment=False):
    mu = mu_2(noise_type, noise_level)
    centre = get_critical_init(noise_type, noise_level)

    sigma_right = (np.finfo(np.float32).max / 1)**(1/n_layers) * (2 / mu)
    sigma_left = (np.finfo(np.float32).tiny / 1)**(1/n_layers) * (2 / mu)

    divisor = 2
    num_samples = 4

    # print(centre)

    right_dists = [(sigma_right - centre) * 0.9]
    for _ in range(num_samples-1):
        right_dists.append(right_dists[-1]/divisor)

    left_dists = [(centre - sigma_left) * 0.9]
    for _ in range(num_samples-1):
        left_dists.append(left_dists[-1]/divisor)

    # print(right_dists)
    # print(left_dists)

    extreme_right = list(centre + np.array(right_dists))[::-1]
    left = list(centre - np.array(left_dists))

    # print(extreme_right)
    # print(left)

    right = list(-(np.array(left) - centre) + centre)[::-1] + extreme_right[-2:]
    # print(right)

    inits = left + [centre] + right
    # print(inits)

    # # shuffle the inits in a way that we get a progressively more fine grained
    # # look at the tests
    # inits = [centre,]
    # pull_from = "end"

    # while len(right) > 0 and len(left) > 0:
    #     if pull_from == "end":
    #         pull_from = "beginning"

    #         inits.append(right[-1])
    #         inits.append(left[-1])

    #         right = right[:-1]
    #         left = left[:-1]
    #     elif pull_from == "beginning":
    #         pull_from = "middle"

    #         inits.append(right[0])
    #         inits.append(left[0])

    #         right = right[1:]
    #         left = left[1:]
    #     elif pull_from == "middle":
    #         pull_from = "end"

    #         index = len(right) // 2

    #         inits.append(right[index])
    #         inits.append(left[index])

    #         right = right[:index] + right[index+1:]
    #         left = left[:index] + left[index+1:]

    if give_boundaries:
        return inits, [sigma_left, sigma_right]

    return inits

def save_model(model, name, epoch, noise_type, noise_level, nan=False, model_dir='../results'):
    hp = hp_dict(noise_type, noise_level)
    name_list = name.split("_")
    exp_indices = [int(name_list[8]), int(name_list[9]), int(name_list[10])]
    noise_type, noise_level, _ = get_experiment(exp_indices, hp["width"][int(name_list[3])]) # hardcoded num_of layers because not using init yet

    # os.makedirs('{}/{}'.format(model_dir, noise_type), exist_ok=True)
    # os.makedirs('{}/{}/{}'.format(model_dir, noise_type, noise_level), exist_ok=True)
    os.makedirs('{}/{}/{}/{}'.format(model_dir, noise_type, noise_level, int(name_list[0])), exist_ok=True)

    save_path = '{}/{}/{}/{}/'.format(model_dir, noise_type, noise_level, int(name_list[0]))

    # # save initialisations
    # init_file_exists = os.path.isfile(save_path + "Initialisations.txt")
    # if not init_file_exists:
    #     init_array = get_initialisations(noise_type, noise_level, hp["width"][int(name_list[2])]) #previously name_list[3] which is incorrect

    #     initial = np.array([])
    #     for i in range(len(init_array)):
    #         initial = np.append(initial, "{}: {}".format(i, init_array[i]))

    #     np.savetxt(save_path + "Initialisations.txt", initial, fmt="%s")

    if nan:
        # save just a text file for NaN
        np.savetxt(save_path + name + '_{}_NaN.txt'.format(epoch), [""], fmt="%s")
        # torch.save(model.state_dict(), save_path + name + '_{}_NaN.pth'.format(epoch))
    else:
        torch.save(model.state_dict(), os.path.join(save_path, "{}_{}.pth".format(name, epoch)))


def model_name(hp, ex):
    name = "{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(hp[0], hp[1], hp[2], hp[3], hp[4], hp[5], hp[6], hp[7], ex[0], ex[1], ex[2])
    return name


def load_model(model_dict, path_to_results="./results", sorted_by_epoch=True):
    '''
    Given the model dict in the form
        {nt: {nt: {exp index: {init index: []}}}

    return the same dictionary with the paths of the given models appended to the list
    '''
    new_model_dict = model_dict

    for nt in new_model_dict: # noise type
        for nl in new_model_dict[nt]: # noise level
            for exp in new_model_dict[nt][nl]: # experiment number
                for i in new_model_dict[nt][nl][exp]: # initialisation
                    path = "{}/{}/{}/{}".format(path_to_results, nt, nl, exp)

                    if os.path.isdir(path):

                        full_list = os.listdir(path)

                        # list of models with no NaNs
                        list_no_txt = [x for x in full_list if x[-4:] != ".txt"]
                        #print(list_no_txt)

                        # list of models with correct initialisation
                        path_list = [path+"/"+x for x in list_no_txt if x.split("_")[-2] == str(i)]

                        # check if it should be sorted by epoch
                        if sorted_by_epoch:
                            sorted_paths = sorted(path_list, key=by_epoch)
                            new_model_dict[nt][nl][exp][i] = sorted_paths
                        else:
                            new_model_dict[nt][nl][exp][i] = path_list

    return new_model_dict

def by_epoch(path):
    # Needed to sort the list of paths by epoch - might not be necessary
    return int(path.split('/')[-1].split("_")[-1].split(".")[0])

def by_exp_number(path):
    return int(path.split('/')[-2])

def by_init(path):
    return int(path.split('/')[-1])


def get_model_id(model_state_path):
    return model_state_path.split('/')[-1][:-4]


def recreate_model(path, dataset="mnist", act="relu"):
    '''
    Return the recreated model from the given model state
    '''
    device, dtype, _ = get_settings()

    file_name = path.split("/")[-1]
    noise_type, noise_level = path.split("/")[-4:-2]

    hyperparams = file_name.split("_")
    hp = []
    for each in hyperparams[:11]:
        hp.append(int(each))

    hyperparams = get_hyperparameters(hp[:8], noise_type, noise_level)

    # batch_size = hyperparams[1]
    n_hidden = hyperparams[3]
    n_layer = hyperparams[2]
    # seed = hyperparams[4]

    # print("Hyperparams", hyperparams)
    # print("n_layer:", n_layer)

    noise_type, noise_level, init_val = get_experiment([hp[8], hp[9], hp[10]], n_layer)


    if noise_type == 'none':
        noise_type = None
        noise_level = None

    # print(noise_type, noise_level, init_val)

    n_in, n_out = get_data_dimensions(dataset)

    model = Net(n_in, n_hidden, n_out, n_layer, act=act, noise_type=noise_type,
          noise_level=noise_level, init_val=init_val).to(device, dtype)

    model.load_state_dict(torch.load(path, map_location=device)) # NB double check that this doesn't break things on GPU
    # model.load_state_dict(torch.load(path, map_location='cpu'))
    # model.eval()

    return model

def get_dir_structure(root_dir = 'results'):
    dir_list = []
    for dirName, subdirList, fileList in os.walk(root_dir):
        # perform check for correct level of depth
        if len(dirName.split('/')) == 4 + len(root_dir.split('/')) -1:
            fname_list = []
            for fname in fileList:
                if fname.split('_')[-1] != 'NaN.txt':
                    try:
                        fname_list.append(fname.split('_')[-2])
                    except:
                        pass
            unique_inits = list(set(fname_list))
            for init in unique_inits:
                dir_list.append(dirName + '/{}'.format(init))
    sorted_dir_list = sorted(dir_list, key=lambda x: (by_exp_number(x), by_init(x)))
    return sorted_dir_list

def get_dict_structure(dir_list):
    # creat dict_list from dir_list
    dir_dict_list = []
    for d in dir_list:
        keys = d.split('/')
        dir_dict = {keys[-4]: {keys[-3]: {keys[-2]: {keys[-1]: []}}}}
        dir_dict_list.append(dir_dict)
    return dir_dict_list

def get_experiment_dicts(root_dir = 'results'):
    dir_list = get_dir_structure(root_dir)
    dir_dict_list = get_dict_structure(dir_list)
    return dir_dict_list

def make_save_path(root_dir, model_id, noise_type, noise_level):
    hp = hp_dict(noise_type, noise_level)
    name_list = model_id.split("_")
    exp_indices = [int(name_list[8]), int(name_list[9]), int(name_list[10])]
    noise_type, noise_level, _ = get_experiment(exp_indices, hp["width"][int(name_list[3])]) # hardcoded num_of layers because not using init yet

    os.makedirs('{}/{}'.format(root_dir, noise_type), exist_ok=True)
    os.makedirs('{}/{}/{}'.format(root_dir, noise_type, noise_level), exist_ok=True)
    os.makedirs('{}/{}/{}/{}'.format(root_dir, noise_type, noise_level, int(name_list[0])), exist_ok=True)

    save_path = '{}/{}/{}/{}/'.format(root_dir, noise_type, noise_level, int(name_list[0]))
    return save_path

def test_models(model_dicts, criterion, train_loader, test_loader,
                result_path='results', dump_path='plotting', quiet=False):
    for model_dict in model_dicts:
        model = load_model(model_dict, result_path)
        for noise_type, nt_dict in model.items():
            for noise_level, nl_dict in nt_dict.items():
                for exp_num, en_dict in nl_dict.items():
                    for init_index, ii_dict in en_dict.items():
                        for model_path in ii_dict:
                            epoch = int(model_path.split('_')[-1].split('.')[0])
                            model = recreate_model(model_path)
                            train_loss, train_acc = test(model, train_loader, criterion, quiet=True)
                            test_loss, test_acc = test(model, test_loader, criterion, quiet=True)
                            if not quiet:
                                print("Testing model: noise_type={}\tnoise_level={}\texp_num={}\tinit_index={}\tepoch={}".format(
                                    noise_type, noise_level, exp_num, init_index, epoch))
                                print("train loss = {}\ttrain accuracy = {}".format(train_loss, train_acc*100))
                                print("test loss = {}\ttest accuracy = {}".format(test_loss, test_acc*100))
                                print()
                            train_path = os.path.join(dump_path, 'train', noise_type, str(noise_level),
                                                      str(exp_num))#, str(init_index))
                            test_path = os.path.join(dump_path, 'test', noise_type, str(noise_level),
                                                     str(exp_num))#, str(init_index))
                            if not os.path.exists(train_path):
                                os.makedirs(train_path)
                            if not os.path.exists(test_path):
                                os.makedirs(test_path)
                            dump_file = "{}.npz".format(get_model_id(model_path))
                            np.savez(os.path.join(train_path, dump_file), loss=[train_loss], accuracy=[train_acc])
                            np.savez(os.path.join(test_path, dump_file), loss=[test_loss], accuracy=[test_acc])

def load_model_results(model_dicts, path_to_results="plotting/test", sorted_by_epoch=True):
    '''
    Given the eval results model dict in the form
        {nt: {nt: {exp index: {init index: []}}}

    return the same dictionary with the .npz result filepaths of the given model appended to the list

    Usage:
    >>> test_model_dicts = get_experiment_dicts(os.path.join('plotting', 'test'))
    >>> load_model_results(test_model_dicts, os.path.join('plotting', 'test'))
    >>> train_model_dicts = get_experiment_dicts(os.path.join('plotting', 'train'))
    >>> load_model_results(train_model_dicts, os.path.join('plotting', 'train'))
    '''
    new_model_dicts = []
    for new_model_dict in model_dicts:
        print(new_model_dict)
        continue

        for nt in new_model_dict: # noise type
            for nl in new_model_dict[nt]: # noise level
                for exp in new_model_dict[nt][nl]: # experiment number
                    path = "{}/{}/{}/{}".format(path_to_results, nt, nl, exp)
                    if os.path.isdir(path):
                        full_list = os.listdir(path)

                        for file_name in full_list:
                            init = file_name.split("_")[-2]
                            # init = int(file_name.split("_")[-2])
                            new_model_dict[nt][nl][exp][init].append(os.path.join(path, file_name))

                        # check if it should be sorted by epoch
                    if sorted_by_epoch:
                        for i in new_model_dict[nt][nl][exp]: # initialisation
                            new_model_dict[nt][nl][exp][i] = sorted(new_model_dict[nt][nl][exp][i], key=by_epoch)

        new_model_dicts.append(new_model_dict)

    exit()
    return new_model_dicts

def load_structured_directory_data(directory, load_data=False, progress_file_name=None, data_file_name=None, force_rescan=False):
    ############################################################################
    # !! new load_data fixes have not been tested !!
    ############################################################################

    if not progress_file_name:
        progress_file_name = "{}_paths.npz".format(directory.strip().split("/")[-1])

    if not data_file_name:
        data_file_name = "{}_data.npz".format(directory.strip().split("/")[-1])

    if load_data and os.path.exists(data_file_name) and not force_rescan:
        print("File exists on disk, simply loading it...")

        data = np.load(data_file_name)

        contents = data.files
        if len(contents) == 1:
            data = data[contents[0]].tolist()

    elif os.path.exists(progress_file_name) and not force_rescan:
        print("File exists on disk, simply loading it...")

        data = np.load(progress_file_name)

        contents = data.files
        if len(contents) == 1:
            data = data[contents[0]].tolist()

        if load_data:
            new_data = {}
            convert_path_dict_to_data(data, new_data)
            data = new_data
            np.savez_compressed(data_file_name, data)

    else:
        data = {}
        directory = os.path.abspath(directory)
        _load_structured_directory_data(directory, data, load_data)
        np.savez_compressed(data_file_name, data)

    return data

def _load_structured_directory_data(directory, data, load_data):
    directory_contents = os.listdir(directory)

    for item in tqdm(directory_contents, ascii=True, desc="Checking {}".format(directory)):
        new_path = os.path.join(directory, item)

        if os.path.isdir(new_path):
            data[item] = {}
            _load_structured_directory_data(new_path, data[item], load_data)

        elif os.path.isfile(new_path): # assume I have reached the final level
            if any(extension in item for extension in [".npz", ".pth", ".npy"]):
                initialisation_index, epoch = item.split(".")[-2].split("_")[-2:]

                if load_data:
                    if initialisation_index not in data:
                        data[initialisation_index] = {}

                    try:
                        temp_data = np.load(new_path)
                    except OSError:
                        temp_data = np.array([np.float("NaN")])
                        print("Failed to load {}".format(new_path))

                    if isinstance(temp_data, np.ndarray):
                        data[initialisation_index][item] = temp_data
                    elif isinstance(temp_data, np.NpzFile):
                        contents = temp_data.files
                        num_items = len(contents)

                        if num_items == 1:
                            data[initialisation_index][epoch] = temp_data[contents[0]]

                        elif num_items > 1:
                            data[initialisation_index][epoch] = {
                                item_key: temp_data[item_key] for item_key in contents
                            }

                        os.unlink(os.path.join('/tmp/rad_data', new_path))
                    else:
                        raise TypeError("Cannot load data of {} file".format(item))

                else:
                    if initialisation_index not in data:
                        data[initialisation_index] = []

                    data[initialisation_index].append(new_path)

        else:
            raise ValueError(
                "`load_structured_directory_data` does not know how to handle file: {}"
                .format(item)
            )

def convert_path_dict_to_data(dictionary, new_dict):
    with tqdm(total=len(dictionary)) as progress_bar:
        for key in list(dictionary):
            progress_bar.set_description(desc="Loading data from {}".format(key))
            item = dictionary[key]

            if isinstance(item, dict):
                new_dict[key] = {}
                convert_path_dict_to_data(item, new_dict[key])

            elif isinstance(item, list):
                for path in tqdm(item):
                    initialisation_index, epoch = path.split(".")[-2].split("_")[-2:]

                    if initialisation_index not in new_dict:
                        new_dict[initialisation_index] = {}

                    temp_data = np.load(path)
                    contents = temp_data.files
                    num_items = len(contents)

                    if num_items == 1:
                        new_dict[initialisation_index][epoch] = temp_data[contents[0]]
                    elif num_items > 1:
                        new_dict[initialisation_index][epoch] = {
                            item_key: temp_data[item_key] for item_key in contents
                        }

                    temp_data.close()

            else:
                raise TypeError("The dictionary passed to `convert_path_dict_to_data` contains invalid entries - all entries must be of type `dict` or `list`")

            progress_bar.update(1)

def get_trajectory_files(trajectory_dir, experiment_name):
    files = os.listdir(trajectory_dir)
    return list(filter(lambda x: ".npy" in x and experiment_name in x, files))

def delete_files(file_dir, files):
    for file_name in files:
        os.remove(os.path.join(file_dir, file_name))

def check_trajectory_files(file_dir, files):
    all_files_present = True
    corrupted_files_present = False

    # check what files are present
    files_present = {
        metric: {"file_name": False, "data": None, "length": None}
        for metric in ["train_accuracy", "test_accuracy", "train_loss", "test_loss"]
    }

    for metric in files_present:
        for file_name in files:
            if metric in file_name:
                # check if file is corrupted
                try:
                    files_present[metric]["data"] = np.load(os.path.join(file_dir, file_name))
                    files_present[metric]["length"] = len(files_present[metric]["data"])
                except OSError:
                    corrupted_files_present = True
                    print("Found corrupted files...")

                files_present[metric]["file_name"] = file_name
                break

    for metric in files_present:
        if not files_present[metric]["file_name"]:
            all_files_present = False
            print("{} file not found, thus test must start from the beginning".format(metric))
            break

    length = None
    if all_files_present and (not corrupted_files_present):
        for metric in files_present:
            if length is None:
                length = files_present[metric]["length"]
                continue
            else:
                if length != files_present[metric]["length"]:
                    corrupted_files_present = True
                    break

    if not all_files_present or corrupted_files_present:
        print("deleting corrupted files...")
        delete_files(file_dir, files)

    return files_present, all_files_present, corrupted_files_present

def check_model_states(model_states_dir, experiment_name):
    NaNed = False
    latest_model = None
    latest_model_epoch = -1

    if os.path.exists(model_states_dir):
        model_states = list(filter(lambda x: any([".pth" in x, ".txt" in x]) and experiment_name in x, os.listdir(model_states_dir)))
        for model_state in model_states:
            if "NaN" in model_state:
                NaNed = True
                break
            else:
                current_epoch = int(model_state.split(".")[-2].split("_")[-1])

                if current_epoch > latest_model_epoch:
                    latest_model_epoch = current_epoch
                    latest_model = os.path.join(model_states_dir, model_state)

    return NaNed, latest_model, latest_model_epoch

def get_train_and_start_epoch(experiment_results_dir, model_states_dir, num_epochs, experiment_name):
    run_train = True
    start_epoch = 0
    model_to_load = None

    # check if there are any files in the experiments dir
    results_files = get_trajectory_files(experiment_results_dir, experiment_name)
    results_present = len(results_files) > 0

    # if there are files, check if they are corrupted (if they are delete the files and start from scratch)
    if results_present:
        print("results directory is not empty, checking what is present")
        file_contents, all_files_present, corrupted_files_present = check_trajectory_files(experiment_results_dir, results_files)

        # if they are not corrupted, check if the file is num_epochs + 1 long
        if all_files_present and (not corrupted_files_present):
            epoch_reached = file_contents["train_accuracy"]["length"] -1
            print(f"training trajectory present until epoch {epoch_reached}")

            if epoch_reached < num_epochs:
                if epoch_reached >= 0:
                    # if it is not, check if train acc reached 100%
                    if file_contents["train_accuracy"]["data"][-1] == 1.0:
                        run_train = False
                    # if not, check if this node contains the model states folder associated with this experiment
                    else:
                        NaNed, latest_model, latest_model_epoch = check_model_states(model_states_dir, experiment_name)
                        # if so, check if this model NaNed
                        if NaNed:
                            # if so, no need for training
                            run_train = False
                        # if not, start from the latest available model state
                        else:
                            if latest_model:
                                model_to_load = latest_model
                                start_epoch = latest_model_epoch

                                print("last model saved at epoch {}".format(start_epoch))
                                # trim training dynamics files
                                for metric in file_contents:
                                    file_path = os.path.join(experiment_results_dir, file_contents[metric]["file_name"])
                                    np.save(file_path, file_contents[metric]["data"][:start_epoch])
                            else:
                                print("No model states found, starting training from scratch...")
                                delete_files(experiment_results_dir, [file_contents[metric]["file_name"] for metric in file_contents])

            else:
                # I could check here if the file is too long, but I should have fixed it so that can't happen anymore
                run_train = False

    return run_train, start_epoch, model_to_load
