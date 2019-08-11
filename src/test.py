import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
# custom import
from src.data_loader import load_data
from src.net import Net
from src.svcca_utils import get_activations, get_diag_correlation
from src.utils import train, test, get_acc


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






def get_correlations(net, model_name, n_epochs, evaluation_epochs, train_loader, test_loader, device):
    criterion = torch.nn.CrossEntropyLoss()
    optimiser = optim.SGD(net.parameters(), lr=0.001)
    trained_net = train(net, train_loader, criterion, optimiser,
                         n_epochs, device, save=True, name=model_name, log_interval=1)
    test(trained_net, test_loader, criterion, device)


    trained_activations = get_activations(net, 'models/{}_{}.pth'.format(model_name, n_epochs - 1), train_loader, device)
    correlations = []
    for epoch in evaluation_epochs:
        print(model_name, 'at epoch:', epoch)
        partial_activations = get_activations(net, 'models/{}_{}.pth'.format(model_name, epoch), train_loader, device)
        correlation = get_diag_correlation(trained_activations, partial_activations)
        correlations.append(np.mean(np.diag(correlation)))
        print("HERE")
    return correlations


def get_score(n_iterations, init_val, model_name, n_epochs, eval_epochs, train_loader, test_loader, device):
    all_correlations = list()
    for i in range(n_iterations):
        net = Net(n_in, n_hidden, n_out, n_layer, act='relu', init_val=init_val).to(device)
        corrs = get_correlations(net, '{}_{}'.format(model_name, i), n_epochs, eval_epochs, train_loader, test_loader, device)
        print(corrs)
        all_correlations.append(corrs)
    Y = np.array(all_correlations)
    # trained_net = train(net, train_loader, criterion, optimiser,
    #                      n_epochs, device, save=True, name=model_name, log_interval=1)
    # test(trained_net, test_loader, criterion, device)y(all_correlations)
    return Y


if __name__ == '__main__':

    # use GPU if available
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(device)
    # set seed
    seed = 1
    torch.manual_seed(seed)

    # set hyperparameters
    n_in = 784
    n_hidden = 100
    n_out = 10
    n_layer = 10
    n_epochs = 20

    train_loader, test_loader = load_data('mnist')

    eval_epochs = np.arange(0, n_epochs)
    eval_epochs = [0, 3, 8, 13, 19]
    n_iterations = 5

    Y1 = get_score(n_iterations, 2, 'init_2', n_epochs, eval_epochs, train_loader, test_loader, device)
    Y2 = get_score(n_iterations, 1, 'init_1', n_epochs, eval_epochs, train_loader, test_loader, device)

    X = np.array(eval_epochs)
    plot_average(X, [Y1, Y2], ['Optimal Init', 'Suboptimal Init'], ['g', 'r'])

