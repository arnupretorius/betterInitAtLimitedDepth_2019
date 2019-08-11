import os, torch
from torchvision import datasets, transforms

def get_data_dimensions(dataset):
    if dataset not in ["mnist", "cifar10", "cifar100", "fashion_mnist"]:
        raise ValueError("Dataset not supported.")
    if dataset == 'mnist':
        return 28*28, 10
    elif dataset == 'cifar10':
        return 32*32*3, 10
    elif dataset == 'cifar100':
        return 32*32*3, 100
    elif dataset == 'fashion_mnist':
        return 28*28, 10

def load_data(dataset='mnist', batch_size=128, shuffle=False, path='./data'):

    if dataset not in ["mnist", "cifar10", "cifar100", "fashion_mnist"]:
        raise ValueError("Dataset not supported.")

    path = os.path.join(path, dataset)

    if batch_size is None:
        batch_size = 99999999999 # load all data into one batch

    transform = transforms.ToTensor()

    if dataset == 'mnist':
        train_dataset = datasets.MNIST(root=path,
                                       train=True,
                                       transform=transform,
                                       download=True)

        test_dataset = datasets.MNIST(root=path,
                                      train=False,
                                      transform=transform)
    elif dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root=path, train=True,
                                         download=True, transform=transform)

        test_dataset = datasets.CIFAR10(root=path, train=False,
                                        download=True, transform=transform)
    elif dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(root=path, train=True,
                                         download=True, transform=transform)

        test_dataset = datasets.CIFAR100(root=path, train=False,
                                        download=True, transform=transform)
    elif dataset == "fashion_mnist":
        train_dataset = datasets.FashionMNIST(root=path, train=True,
                                         download=True, transform=transform)

        test_dataset = datasets.FashionMNIST(root=path, train=False,
                                        download=True, transform=transform)


    # Data Loader (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=shuffle)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle)

    return train_loader, test_loader
