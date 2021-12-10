import tensorflow as tf
import tensorflow_datasets as tfds

# 'unlabeled'            ,  0
# 'ego vehicle'          ,  1
# 'rectification border' ,  2
# 'out of roi'           ,  3
# 'static'               ,  4
# 'dynamic'              ,  5
# 'ground'               ,  6
# 'road'                 ,  7
# 'sidewalk'             ,  8
# 'parking'              ,  9
# 'rail track'           , 10
# 'building'             , 11
# 'wall'                 , 12
# 'fence'                , 13
# 'guard rail'           , 14
# 'bridge'               , 15
# 'tunnel'               , 16
# 'pole'                 , 17
# 'polegroup'            , 18
# 'traffic light'        , 19
# 'traffic sign'         , 20
# 'vegetation'           , 21
# 'terrain'              , 22
# 'sky'                  , 23
# 'person'               , 24
# 'rider'                , 25
# 'car'                  , 26
# 'truck'                , 27
# 'bus'                  , 28
# 'caravan'              , 29
# 'trailer'              , 30
# 'train'                , 31
# 'motorcycle'           , 32
# 'bicycle'              , 33
# 'license plate'        , -1

# fmt: off
classes = {
    7: 1,   # road
    26: 2,  # car
    24: 3   # person
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


def create_cityscapes_ds(split, batch_size):
    ds = tfds.load(
        "cityscapes", data_dir="tensorflow_datasets", download=False, split=split
    )
    ds = ds.map(preproc, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.shuffle(100)
    # ds = ds.take(1 * batch_size)
    # ds = ds.repeat(2)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    # ds = ds.take(1).repeat(1000)
    return ds
