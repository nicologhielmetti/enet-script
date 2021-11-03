import tensorflow as tf
import tensorflow_datasets as tfds
import yaml
import numpy as np

classes = {
    7: 1,  # road
    26: 2,  # car
    24: 3  # person
}
# fmt: on

N_CLASSES = len(classes.keys()) + 1
WIDTH = 152
HEIGHT = 240
CROP_FRAC = 0.05
BOX_FRAC = 0.8


def preproc(data):
    # box_starts = [[CROP_FRAC, CROP_FRAC, 1.0 - CROP_FRAC, 1.0 - CROP_FRAC]]
    # box_widths = tf.random.uniform(shape=(1, 4), minval=-CROP_FRAC, maxval=CROP_FRAC)
    box_starts = tf.random.uniform(shape=(1, 2), minval=0, maxval=(1.0 - BOX_FRAC))
    boxes = tf.concat([box_starts, box_starts + [[BOX_FRAC, BOX_FRAC]]], axis=-1)
    box_idx = [0]
    # image = tf.image.resize(data["image_left"], (WIDTH, HEIGHT)) / 255.0
    # segmentation = tf.image.resize(data["segmentation_label"], (WIDTH, HEIGHT), method="nearest")
    image = (
            tf.image.crop_and_resize(
                tf.expand_dims(data["image_left"], 0),
                boxes,
                box_idx,
                crop_size=(HEIGHT, WIDTH),
            )
            / 255.0
    )
    segmentation = tf.image.crop_and_resize(
        tf.expand_dims(data["segmentation_label"], 0),
        boxes,
        box_idx,
        crop_size=(HEIGHT, WIDTH),
        method="nearest",
    )
    image = tf.squeeze(image, 0)
    segmentation = tf.squeeze(segmentation, 0)
    segmentation = tf.cast(segmentation, tf.int32)
    output_segmentation = tf.zeros(segmentation.shape, dtype=segmentation.dtype)

    for cs_class, train_class in classes.items():
        output_segmentation = tf.where(
            segmentation == cs_class, train_class, output_segmentation
        )

    # image = tf.transpose(image, [2, 0, 1])
    # segmentation = tf.transpose(segmentation, [2, 0, 1])
    return image, output_segmentation


def create_cityscapes_ds(split, batch_size, n_elem):
    ds = tfds.load(
        "cityscapes", data_dir="tensorflow_datasets", download=True, split=split
    )
    ds = ds.map(preproc, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.shuffle(100)
    ds = ds.take(n_elem)
    # ds = ds.repeat(2)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    # ds = ds.take(1).repeat(1000)
    ds_numpy = tfds.as_numpy(ds)
    x_n_elem_ds = []
    y_n_elem_ds = []
    for elem in ds_numpy:
        x_n_elem_ds.append(elem[0][0])
        y_n_elem_ds.append(elem[1][0])
    np.save('X_' + split + '.npy', x_n_elem_ds)
    np.save('y_' + split + '.npy', y_n_elem_ds)


def read_dataset(path):
    return np.load(path)


create_cityscapes_ds('test', 1, 10)
