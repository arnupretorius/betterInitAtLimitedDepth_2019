import torch
import tables
import os
import numpy as np

# custom imports
from src.cca_core import get_cca_similarity, robust_cca_similarity
from src.utils import get_settings, recreate_model, get_model_id, make_save_path, get_experiment, exp_dict


def activation_hook(store, func):
    return lambda module, input, output: func(store, output.cpu().numpy())


def to_memory_hook(activations, output):
    activations.append(output)


def to_file_hook(file, output):
    if not hasattr(file.root, 'data'):
        atom = tables.Float64Atom()
        shape = list(output.shape)
        shape[0] = 0
        file.create_earray(file.root, 'data', atom, tuple(shape))
    for row in output:
        file.root.data.append(row[np.newaxis, ...])


def get_activations(model, path, loader, layers="all"):
    device, dtype, dtype_y = get_settings()
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()

    activations = []
    all_layers = list(model.layers)
    if layers == 'all':
        selected_layers = all_layers
    else:
        selected_layers = [all_layers[l] for l in layers]

    for l, layer in enumerate(selected_layers):
        activations.append([])
        layer.register_forward_hook(
            activation_hook(activations[l], to_memory_hook))

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.to(device, dtype), target.to(device, dtype_y)
            model(data)

    activations_cat = []
    for activation in activations:
        activations_cat.append(torch.cat(activation, dim=0).cpu().numpy().T)
    return activations_cat


def get_correlation_all_permutations(a, b):
    num_layer = len(a)
    correlation = np.zeros((num_layer, num_layer))
    for i, a1 in enumerate(a):
        for j, a2 in enumerate(b):
            cca = get_cca_similarity(
                a1, a2, compute_dirns=False, verbose=False)
            if i <= j:
                correlation[i, j] = cca['mean'][0]
                correlation[j, i] = cca['mean'][0]
    return correlation


# Note: this function doesn't get used anywhere?
def get_diag_correlation(a, b):
    num_layer = len(a)
    correlation = np.zeros((num_layer, num_layer))
    for i, a1 in enumerate(a):
        a2 = b[i]
        cca = get_cca_similarity(a1, a2, compute_dirns=False, verbose=False)
        correlation[i, i] = cca['mean'][0]
    return correlation


def write_activations_to_file(model, save_path, data_loader, layers):
    device, dtype, dtype_y = get_settings()
    files = list()
    handles = list()

    for i, layer in enumerate(layers):
        os.makedirs(save_path, exist_ok=True)
        file = tables.open_file(
            '{}/layer_{}.h5'.format(save_path, i), mode='w')
        files.append(file)
        handle = layer.register_forward_hook(
            activation_hook(file, to_file_hook))
        handles.append(handle)

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data, target = data.to(
                device, dtype), target.to(device, dtype_y)
            model(data)

    for file in files:
        file.close()

    for handle in handles:
        handle.remove()


def dump_model_state_activations(model_state_path, data_loader, save_path, final=False):

    print('-------------------------------')
    print('GETTING MODEL STATE ACTIVATIONS')
    print('-------------------------------')
    device, dtype, dtype_y = get_settings()

    if final:
        save_path = save_path + '/model_final_act'
    else:
        save_path = save_path + '/model_act'

    net = recreate_model(model_state_path)
    # net.load_state_dict(torch.load(model_state_path, map_location=device))
    # net.to(device)
    net.eval()
    write_activations_to_file(net, save_path, data_loader, list(net.layers))


def get_diag_svcca_correlations(correlations_path, epochs):
    correlations = np.load('{}/correlations.npz'.format(correlations_path))
    correlations = [np.diag(correlations['arr_{}'.format(epoch)]) for epoch in range(epochs)]
    return correlations


def save_correlations(save_path, model_id, correlations):
    save_path = local_make_save_path(save_path, model_id)
    np.savez('{}/{}_correlations.npz'.format(save_path, model_id), correlations)


def get_svcca_correlations_per_layer(model_state_path, path, save_path, final=False, projection_weighted=False):

    print('--------------------------')
    print('GETTING SVCCA CORRELATIONS')
    print('--------------------------')

    model_id = get_model_id(model_state_path)
    #net = recreate_model(model_state_path) # this is a pretty slow way of getting the number of layers....
    #n_layers = len(list(net.layers)) # we should just be checking the hyper-params wit the utils.py functions
    int_id_list = [int(x) for x in model_id.split('_')]
    n_layers = get_hyperparameters(int_id_list)[2]

    print('Number of layers: {}'.format(n_layers))

    correlations = np.zeros((n_layers, n_layers))

    for layer in range(0, n_layers):
        trained_activation_path = '{}/model_final_act'.format(path)
        print(trained_activation_path)
        f = tables.open_file('{}/layer_{}.h5'.format(trained_activation_path, layer))
        trained_activations = np.asarray(f.root.data).T
        print(trained_activations)
        f.close()
        if final:
            activation_path = '{}/model_final_act'.format(path)
        else:
            activation_path = '{}/model_act'.format(path)
        print(activation_path)
        for other_layer in range(0, n_layers):
            if layer <= other_layer:
                print('Comparing layer', layer, 'to', other_layer)
                f = tables.open_file('{}/layer_{}.h5'.format(activation_path, other_layer))
                partial_activations = np.asarray(f.root.data).T
                print(partial_activations)
                f.close()
                if projection_weighted:
                    pwcca_mean,_,_ = compute_pwcca(trained_activations, partial_activations)
                    print(pwcca_mean)
                    correlations[layer, other_layer] =  pwcca_mean
                    correlations[other_layer, layer] =  pwcca_mean
                else:
                    cca = robust_cca_similarity(trained_activations, partial_activations)
                    print(cca['mean'])
                    print(cca['mean'][0])
                    correlations[layer, other_layer] = cca['mean'][0]
                    correlations[other_layer, layer] = cca['mean'][0]

    save_correlations(save_path, model_id, correlations)

def local_make_save_path(save_path, model_id):
        exps = exp_dict()
        name_list = model_id.split("_")
        exp_indices = [int(name_list[8]), int(name_list[9]), int(name_list[10])]
        noise_type = list(exps.keys())[exp_indices[0]]
        noise_level = exps[noise_type][exp_indices[1]]
        base_path = make_save_path(save_path, model_id, noise_type, noise_level)
        return base_path

def dump_svcca_correlations(model_state_paths, train_loader, n_slices=5, act_path='temp_activations', corr_path='correlations', projection_weighted=False):

    print(n_slices)
    if isinstance(n_slices, int):
        final_epoch = len(model_state_paths)-1
        epochs = np.linspace(0, final_epoch, n_slices, dtype=int)[:-1]
    else:
        final_epoch = n_slices[-1]
        epochs = n_slices[:-1]

    print(final_epoch)
    print(epochs)
    print(model_state_paths)
    # get final epoch correlation for every layer
    final_model_state = model_state_paths[final_epoch]
    print(final_model_state)
    dump_model_state_activations(final_model_state, train_loader, act_path, final=True)
    get_svcca_correlations_per_layer(final_model_state, act_path, corr_path, final=True, projection_weighted=projection_weighted)

    # get correlations at other specified epochs
    for epoch in epochs:
        print('--------------------------')
        print("Getting SVCCA for epoch {}".format(epoch))
        print('--------------------------')

        model_state = model_state_paths[epoch]
        model_id = get_model_id(model_state)
        base_path = local_make_save_path(corr_path, model_id)
        save_file_name = '{}/{}_correlations.npz'.format(base_path, model_id)

        #if os.path.exists(save_file_name):
        #    print('SVCCA FOR THIS EPOCH HAS ALREADY BEEN COMPUTED. THUS, SKIPPING THIS EPOCH.')
        #    continue

        dump_model_state_activations(model_state, train_loader, act_path)
        get_svcca_correlations_per_layer(model_state, act_path, corr_path, projection_weighted=projection_weighted)



