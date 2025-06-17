import mlx.core as mx
import numpy as np
from mlx.data.datasets import load_cifar10
from mlx.data.datasets import load_images_from_folder
from mlx.data import stream_python_iterable
from mlx.data import buffer_from_vector


def get_cifar10(batch_size, root=None):
    tr = load_cifar10(root=root)

    mean = np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3))
    std = np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3))

    def normalize(x):
        x = x.astype("float32") / 255.0
        return (x - mean) / std

    group = mx.distributed.init()

    tr_iter = (
        tr.shuffle()
        .partition_if(group.size() > 1, group.size(), group.rank())
        .to_stream()
        .image_random_h_flip("image", prob=0.5)
        .pad("image", 0, 4, 4, 0.0)
        .pad("image", 1, 4, 4, 0.0)
        .image_random_crop("image", 32, 32)
        .key_transform("image", normalize)
        .batch(batch_size)
        .prefetch(4, 4)
    )

    test = load_cifar10(root=root, train=False)
    test_iter = (
        test.to_stream()
        .partition_if(group.size() > 1, group.size(), group.rank())
        .key_transform("image", normalize)
        .batch(batch_size)
    )

    return tr_iter, test_iter

def get_flower(batch_size, root=None):
    list_flower = list(load_images_from_folder(root).shuffle().to_stream())
    prop_train = 8
    prop_test = 2
    prop_total = prop_train + prop_test
    num_train = int(len(list_flower) * prop_train / prop_total)
    list_flower_train = list_flower[:num_train]
    list_flower_test  = list_flower[num_train:]
    buff_flower_train = buffer_from_vector(list_flower_train)
    buff_flower_test  = buffer_from_vector(list_flower_test)

    mean = np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3))
    std = np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3))

    def normalize(x):
        x = x.astype("float32") / 255.0
        return (x - mean) / std

    group = mx.distributed.init()

    train_iter = (
        buff_flower_train.partition_if(group.size() > 1, group.size(), group.rank())
          .to_stream()
          .image_random_h_flip("image", prob=0.5)
          .pad("image", 0, 4, 4, 0.0)
          .pad("image", 1, 4, 4, 0.0)
          .image_random_crop("image", 32, 32)
          .key_transform("image", normalize)
          .batch(batch_size)
          .prefetch(4, 4)
    )

    test_iter = (
        buff_flower_test.to_stream()
          .partition_if(group.size() > 1, group.size(), group.rank())
          .key_transform("image", normalize)
          .batch(batch_size)
    )

    return train_iter, test_iter
