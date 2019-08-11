import os, torch
import numpy as np

from torchvision import transforms
from src.manifold_exploration import gen_path

class SensitivityTest(torch.utils.data.Dataset):
    data_file = "data.pt"
    def __init__(self, root, data_points, classes, num_interpolations, image_shape, transform=None, re_generate=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.image_shape = image_shape

        self._gen_data(data_points, classes, num_interpolations, re_generate)
        self.data, self.targets = torch.load(os.path.join(self.processed_folder, self.data_file))

    def __getitem__(self, index):
        img = self.data[index].reshape(self.image_shape).transpose((1, 2, 0))

        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.data)

    def _gen_data(self, data_points, classes, num_interpolations, re_generate):
        if re_generate or not self._check_exists():
            # resolution = int(np.round(np.sqrt(num_interpolations + 0.25) + 0.5))
            resolution = num_interpolations // len(classes)
            raw_data, mixture_coefficients, _ = gen_path(data_points, resolution, len(classes))

            targets = self._mixtures_to_distributions(10, classes, mixture_coefficients)

            os.makedirs(self.processed_folder, exist_ok=True)

            with open(os.path.join(self.processed_folder, self.data_file), 'wb') as f:
                torch.save((raw_data, targets), f)

    def _mixtures_to_distributions(self, num_classes, classes, mixtures):
        distributions = np.zeros((mixtures.shape[0], num_classes))
        distributions[:, np.array(classes)] = mixtures

        return distributions

    @property
    def processed_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'processed')

    def _check_exists(self):
        return os.path.exists(os.path.join(self.processed_folder, self.data_file))

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
