import torch
import numpy as np

from src import sensitivity
from src.utils import get_settings, get_acc
from torchvision import transforms
from src.data_loader import load_data
from src.sensitivity import local_sensitivity
from src.sensitivity_dataset import SensitivityTest

def get_sensitivity_data_loader(dataset, classes, data_path, num_interpolations, re_generate=False):
    if not isinstance(classes, list):
        raise ValueError("Classes must be a list of integers")
    elif not isinstance(classes[0], int):
        raise ValueError("Classes must be a list of integers")

    train_loader, _ = load_data(dataset, path=data_path, batch_size=None)

    data = None
    lables = None

    # put code in here to calculate means and store all sorted classes?
    for batch, target in train_loader:
        image_shape = batch.shape[1:]

        data = batch.view(batch.shape[0], -1)
        labels = target

    if re_generate:
        print("regenerate data")
        sorted_data = {i: data[labels == i] for i in range(10)}
        means = {i: sorted_data[i].mean(dim=0) for i in range(10)}

        ############################################################################
        # dist = np.linalg.norm((data[:, :, np.newaxis] - means), axis=1)
        ############################################################################

        for i in range(10):
            # find data point for each class that is closest to the mean (order them?)
            images = sorted_data[i]
            mean = means[i].unsqueeze(0)

            difference = images - mean
            distances = torch.norm(difference, dim=-1)
            order = np.argsort(distances)
            sorted_data[i] = sorted_data[i][order]

        data_points = [sorted_data[_class][0].numpy() for index, _class in enumerate(classes)]
    else:
        data_points = None

    # plot_means_data_points([means[c] for c in classes], data_points, image_shape)

    st = SensitivityTest(
        root=data_path, data_points=data_points, classes=classes, re_generate=re_generate,
        num_interpolations=num_interpolations, transform=transforms.ToTensor(), image_shape=image_shape
    )

    # plot_targets(st.targets, classes)

    return torch.utils.data.DataLoader(dataset=st, batch_size=1, shuffle=False)

# ################################################################################
# # Test function
# ################################################################################
# import matplotlib.pyplot as plt

# def plot_means_data_points(means, data_points, image_shape):
#     num_means = len(means)
#     fig, [means_axes, data_points_axes] = plt.subplots(nrows=2, ncols=num_means, figsize=(4*num_means, 10))

#     for i in range(num_means):
#         ax = means_axes[i]
#         mean = means[i]
#         mean -= mean.min()
#         mean /= mean.max()

#         ax.imshow(mean.reshape(image_shape).permute(1, 2, 0))
#         # ax.imshow(mean.reshape((28, 28)))

#     for i in range(num_means):
#         ax = data_points_axes[i]
#         data_point = data_points[i]
#         data_point -= data_point.min()
#         data_point /= data_point.max()
#         ax.imshow(data_point.reshape(image_shape).transpose(1, 2, 0))
#         # ax.imshow(data_point.reshape((28, 28)))

#     plt.savefig("means_data_points.pdf")
#     # plt.show()

# def plot_targets(targets, classes):
#     plt.figure(figsize=(15, 10))

#     for c in classes:
#         plt.plot(targets[:, c], label=c)

#     num_classes = len(classes)
#     plt.xticks(np.linspace(0, targets.shape[0], num_classes+1), classes + [classes[0]])
#     plt.legend()

#     plt.savefig("targets_points.pdf")

# ################################################################################
# ################################################################################

# def check_accuracy(model):
#     train_loader, _ = load_data('mnist', path="../data")
#     accuracy = get_acc(model, train_loader)
#     return accuracy

def get_data(dataset, data_path, mode="inter_class", re_generate=False):
    if mode == "intra_class":
#         classes = [1, 1, 1]
        classes = [0, 0, 0]
    elif mode == "inter_class":
        # classes = [4, 5, 7]
        classes = [4, 5, 7, 1, 9, 3, 8, 0, 6, 2]
#         classes = [1, 7, 9]
    else:
        raise ValueError("invalid value for the 'mode' argument")

    return get_sensitivity_data_loader(dataset, classes, data_path, num_interpolations=1000, re_generate=re_generate)

def test_sensitivity(model, input_data_loader):
    # set to gpu if available and set dtype to float32
    device, dtype, _ = get_settings()

    sensitivities = []
    for batch in input_data_loader:
        sensitivities.append(local_sensitivity(batch.to(device, dtype), model).detach().cpu().numpy())

    return np.array(sensitivities)

def get_output_distributions(model, input_data_loader):
    # set to gpu if available and set dtype to float32
    device, dtype, _ = get_settings()

    dists = []
    for batch in input_data_loader:
        prediction = model.predict(batch.to(device, dtype)).detach().cpu().numpy()
        dists.extend(prediction)

    return np.array(dists).squeeze()

# def test_model(model, data_loader):
#     # Check model's train accuracy first...
#     check_accuracy(model)

#     # test sensitivity
#     sens = test_sensitivity(model, data_loader)

#     # get predictions from the model
#     dists = get_output_distributions(model, data_loader)

#     plot_test_results(sens, dists, data_loader.dataset.targets)

# def get_avg_sensitivity(model, test_loader):
#     device, dtype, dtype_y = get_settings()
#     model.eval()
#     local_sensitivity = 0
#     for data, target in test_loader:
#         data, _ = data.to(device, dtype), target.to(device, dtype_y)
#         local_sensitivity += sensitivity.local_sensitivity(data, model).data
#     local_sensitivity /= len(test_loader.dataset)
#     return local_sensitivity