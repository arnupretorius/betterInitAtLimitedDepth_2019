import sys, os
from multiprocessing import Pool

file_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(file_dir, ".."))

from src.utils import load_data

def local_load_data(dataset):
    print("checking if {} is present...".format(dataset))
    path = os.path.join(file_dir, "../data")
    load_data(dataset, path=path)
    print("Confirmed: {} is present.".format(dataset))

if __name__ == "__main__":
    with Pool() as p:
        p.map(local_load_data, ["mnist", "cifar10", "cifar100", "fashion_mnist"])
