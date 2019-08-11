import sys, os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.sensitivity_utils import get_data

if __name__ == "__main__":
    dataset = sys.argv[1]
    data_path = sys.argv[2]
    sensitivity_data = get_data(dataset, data_path=data_path, mode="inter_class", re_generate=True)
    # sensitivity_data = get_data(dataset, mode="intra_class", re_generate=True)
