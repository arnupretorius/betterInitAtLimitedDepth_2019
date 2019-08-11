import matplotlib.pyplot as plt

import numpy as np



def plot_average(X, YS, names, colours):

    plt.figure()
    plt.title("")
    plt.xlabel("Epoch")
    plt.ylabel("Average Correlation")
    plt.grid()

    for Y, name, colour in zip(YS, names, colours):

        score_mean = np.mean(Y, axis=0)
        score_std = np.std(Y, axis=0)
        plt.fill_between(X, score_mean - score_std,
                         score_mean + score_std, alpha=0.1,
                         color=colour)
        plt.plot(X, score_mean, 'o-', color=colour,
                 label=name)
    plt.legend(loc="best")


    plt.show()


if __name__ == '__main__':


    seed = 42
    init = 2
    epochs = 20
    n_layers = 11
    path = '../models/mnist/seed_{}_init_{}'.format(seed, init)

    correlations = np.load('{}/correlations.npz'.format(path))
    correlations = [correlations['arr_{}'.format(epoch)] for epoch in range(epochs)]


    fig, grid = plt.subplots(4, 5, sharex='col', sharey='row', figsize=(20, 10))

    for correlation, ax in zip(correlations, grid.ravel()):
        im = ax.imshow(correlation, vmin=0, vmax=1)

    seed = 42
    init = 1
    epochs = 20
    n_layers = 11
    path = '../models/mnist/seed_{}_init_{}'.format(seed, init)

    correlations = np.load('{}/correlations.npz'.format(path))
    correlations = [correlations['arr_{}'.format(epoch)] for epoch in range(epochs)]


    fig, grid = plt.subplots(4, 5, sharex='col', sharey='row', figsize=(20, 10))

    for correlation, ax in zip(correlations, grid.ravel()):
        im = ax.imshow(correlation, vmin=0, vmax=1)

    # plt.show()


    seeds = [42, 43, 44]


    inits = [2, 1]
    correlation_scores = [[], []]
    for i, init in enumerate(inits):
        for seed in seeds:
            path = '../models/mnist/seed_{}_init_{}'.format(seed, init)
            correlations = np.load('{}/correlations.npz'.format(path))
            correlations = [correlations['arr_{}'.format(epoch)] for epoch in range(epochs)]
            avg_diag = [np.mean(np.diag(correlation)) for correlation in correlations]
            correlation_scores[i].append(avg_diag)

    X = np.arange(0, epochs)
    Y1 = np.array(correlation_scores[0])
    Y2 = np.array(correlation_scores[1])

    plot_average(X, [Y1, Y2], ['Optimal Init', 'Suboptimal Init'], ['g', 'r'])





