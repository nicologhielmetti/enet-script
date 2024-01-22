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
WIDTH = 240
HEIGHT = 152
CROP_FRAC = 0.05
BOX_FRAC = 0.8
scaling = 256.0


def get_preproc(seed):
    def preproc(data):
        # box_starts = [[CROP_FRAC, CROP_FRAC, 1.0 - CROP_FRAC, 1.0 - CROP_FRAC]]
        # box_widths = tf.random.uniform(shape=(1, 4), minval=-CROP_FRAC, maxval=CROP_FRAC)
        if seed is not None:
            tf.random.set_seed(seed)
        box_starts = tf.random.uniform(shape=(1, 2), minval=0, maxval=(1.0 - BOX_FRAC), seed=seed)
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
                method="nearest"
            )
            / 256.0
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

    return preproc

def create_cityscapes_ds(split, batch_size, shuffle=True, seed=None):
    ds = tfds.load(
        "cityscapes", data_dir="tensorflow_datasets", download=True, split=split
    )
    ds = ds.map(get_preproc(seed), num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(100)
    # ds = ds.take(1 * batch_size)
    # ds = ds.repeat(2)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    # ds = ds.take(1).repeat(1000)
    return ds


#  create_cityscapes_ds('test', 1, 10)
