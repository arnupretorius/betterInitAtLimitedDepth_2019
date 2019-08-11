import matplotlib.pyplot as plt
import numpy as np

# custom imports
from src.svcca_utils import get_diag_svcca_correlations
from src.utils import get_model_id

def svcca_heatmap(model_state_paths, correlations_path, epochs, nrow=1, ncol=1, save=False, save_path='plot', **kwargs):
    
    correlations = []
    for epoch in epochs:
        model_state = model_state_paths[epoch]
        model_id = get_model_id(model_state)
        correlation = np.load('{}/{}_correlations.npz'.format(correlations_path, model_id))
        correlations.append(correlation['arr_0'])

    # need to make these plotting parameters arguments to the function based on the number of epochs being compared
    fig, grid = plt.subplots(nrow, ncol, sharex='col', sharey='row', **kwargs)

    for correlation, ax in zip(correlations, grid.ravel()):
        im = ax.imshow(correlation, vmin=0, vmax=1)
    if save:
        plt.savefig(save_path)
    else:
        plt.show()

def svcca_average(model_state_paths, correlations_path, epochs, colour='black', name=None, save=False, save_path='plot'):
    
    diag_correlations = []
    for epoch in epochs:
        model_state = model_state_paths[epoch]
        model_id = get_model_id(model_state)
        correlation = np.load('{}/{}_correlations.npz'.format(correlations_path, model_id))
        diag_correlations.append(np.diag(correlation['arr_0']))

    score_mean = np.array([np.mean(correlation) for correlation in diag_correlations])
    score_std = np.array([np.std(correlation) for correlation in diag_correlations])
    plt.fill_between(epochs, score_mean - score_std,
                         score_mean + score_std, alpha=0.1,
                         color=colour)
    plt.plot(epochs, score_mean, 'o-', color=colour, label=name)
    if save:
        plt.savefig(save_path)
    else:
        plt.show()

    


