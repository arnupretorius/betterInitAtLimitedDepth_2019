import numpy as np

def gen_vec(phi, theta):
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    x = sin_theta * cos_phi
    y = sin_theta * sin_phi
    z = cos_theta

    return (x, y, z)

def angle_between(a, b):
    return np.arccos(np.inner(a, b))

def get_data_coords(num_data_points):
    return np.vstack([np.eye(3), -np.eye(3)[:num_data_points-3]])

def get_angles_to_data_points(vectors, resolution, num_data_points=3):
    # I am sure I can vectorize the for loops out but that would require not using the angle_between function
    angles = np.empty((resolution * (resolution - 1) + 1, num_data_points))
    data_point_coords = get_data_coords(num_data_points)

    for vector_index, vector in enumerate(vectors):
        for data_index, data_point in enumerate(data_point_coords):
            angles[vector_index, data_index] = angle_between(vector, data_point)

    return angles, data_point_coords

def rank_angles(angles):
    return np.argsort(angles)

def create_sub_path_sets(vectors, angles, data_point_coords):
    ranking = rank_angles(angles)

    num_data_points = len(data_point_coords)
    vector_sets = {
        (index, (index + 1) % num_data_points): [] for index, _ in enumerate(data_point_coords)
    }
    angle_sets = {
        (index, (index + 1) % num_data_points): [] for index, _ in enumerate(data_point_coords)
    }
    for i, rank in enumerate(ranking):
        key = tuple(rank[0:2])

        if key[0] == (key[1] + 1) % len(data_point_coords):
            key = tuple(np.roll(key, 1))

        vector_sets[key].append(vectors[i])
        angle_sets[key].append(angles[i][key[0]])

    return vector_sets, angle_sets

def order_sub_paths(vector_sets, angle_sets):
    path = []

    for key, vector_set in vector_sets.items():
        angle_set = angle_sets[key]
        order = np.argsort(angle_set)
        path.extend(np.array(vector_set)[order])

    return path

def path_to_data(vector_path, data_points):
    data = []

######### TRY REMOVE FOR LOOP #################

    for vector in vector_path:
        data.append(np.sum(vector[:, np.newaxis] * data_points, axis=0))

    return np.array(data) # if for loop is removed, remove np.array

def smooth(vectors):
    # np_vectors = np.array(vectors)
    # smoothed_vectors = np.empty(np_vectors.shape)
    smoothed_vectors = np.array(vectors)

    # window_sizes = np.arange(2, 50, 5)

    # for window_size in window_sizes:
    #     for i in np.arange(smoothed_vectors.shape[0]):
    #         smoothed_vectors[i] = smoothed_vectors[i : i + window_size].mean(axis=0)

    # smoothed_vectors[0] = smoothed_vectors[0]
    # for i in np.arange(smoothed_vectors.shape[0] - 1):
    #     smoothed_vectors[i+1] = 0.95 * smoothed_vectors[i] + 0.05 * smoothed_vectors[i+1]

    return smoothed_vectors

def gen_path(data_points, resolution, num_classes):
    # num_points = 333
# num_points_last_class = total - len(classes) * num_points

    #######################################################

    zero_to_one = np.linspace(0, 1, resolution)
    one_to_zero = 1 - zero_to_one

    vectors = np.zeros((resolution * num_classes, num_classes))

    for i in range(num_classes):
        start_index = resolution * i
        end_index = resolution * (i + 1)
        vectors[start_index:end_index, i] = one_to_zero
        vectors[start_index:end_index, (i+1) % num_classes] = zero_to_one

    # vectors[:resolution, 0] = one_to_zero
    # vectors[:resolution, 1] = zero_to_one

    # vectors[resolution:resolution*2, 1] = one_to_zero
    # vectors[resolution:resolution*2, 2] = zero_to_one

    # vectors[resolution*2:resolution*3, 2] = one_to_zero
    # vectors[resolution*2:resolution*3, 0] = zero_to_one


    # vectors = np.cosh(vectors)
    # vectors -= np.min(vectors, axis=0)
    # vectors = np.divide(vectors, np.sum(vectors, axis=-1)[:, np.newaxis])

    #######################################################

    # vectors = []
    # for phi in np.linspace(0, np.pi/2, resolution):
    #     for theta in np.linspace(np.pi/2, 0, resolution - 1, endpoint=False):
    #         vectors.append(gen_vec(phi, theta))

    # vectors.append(gen_vec(0, 0))

    # angles, data_point_coords = get_angles_to_data_points(vectors, resolution)
    # vector_sets, angle_sets = create_sub_path_sets(vectors, angles, data_point_coords)

    # vector_path = order_sub_paths(vector_sets, angle_sets)

    # # print(vector_path)
    # # print(np.linalg.norm(vector_path, axis=-1))

    # # print("============================== Sum of Original =============================")
    # # print(np.sum(vector_path, axis=-1))

    # smoothed_path = smooth(vector_path)

    # new_vectors = np.divide(smoothed_path, np.sum(vector_path, axis=-1)[:, np.newaxis])

    # print("============================== Scaled =============================")
    # print(new_vectors)
    # print("============================== Sum of Scaled =============================")
    # print(np.sum(new_vectors, axis=-1))
    # print("============================== Norm of Scaled =============================")
    # print(np.linalg.norm(new_vectors, axis=-1))

    # exp_vectors = np.exp(vector_path)
    # soft_max_vectors = np.divide(exp_vectors, np.sum(exp_vectors, axis=1)[:, np.newaxis])
    # print("============================== Softmax =============================")
    # print(soft_max_vectors)
    # print("============================== Sum of Softmax =============================")
    # print(np.sum(soft_max_vectors, axis=-1))
    # print("============================== Norm of Softmax =============================")
    # print(np.linalg.norm(soft_max_vectors, axis=-1))

    new_data = path_to_data(vectors, data_points)
    # new_data = path_to_data(new_vectors, data_points)

    # targets = to_dist(classes=10, used_classes=, values=new_vectors)

    return new_data, vectors, None
    # return new_data, np.array(new_vectors), vector_sets
    # return new_data, targets, vector_sets