import argparse
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import json
import qkeras
import hls4ml

from cityscapes import create_cityscapes_ds, N_CLASSES, WIDTH, HEIGHT

CHANNEL_AXIS = -1  # 1


class Evaluator:
    def __init__(self, num_classes):
        self.iou = tf.keras.metrics.MeanIoU(num_classes=num_classes)
        self.accuracy = tf.keras.metrics.Accuracy()

    def add_sample(self, prediction, label):
        argmax = np.argmax(prediction, axis=-1)
        self.iou.update_state(np.reshape(label, -1), np.reshape(argmax, -1))
        self.accuracy.update_state(np.reshape(label, -1), np.reshape(argmax, -1))
        return self.result()

    def result(self):
        return dict(
                miou=float(self.iou.result().numpy()),
                accuracy=float(self.accuracy.result().numpy())
                )


def eval_keras_model(model_path):
    model = tf.keras.models.load_model(model_path, compile=False)
    data = create_cityscapes_ds("validation", 1)

    evaluator = Evaluator(N_CLASSES)

    for image, label in tqdm(data):
        prediction = model(image)
        evaluator.add_sample(prediction.numpy(), label.numpy())

    return evaluator.result()

def eval_qkeras_model(model_path):
    model = qkeras.utils.load_qmodel(model_path, compile=False)
    qkeras.utils.model_save_quantized_weights(model)
    data = create_cityscapes_ds("validation", 1)

    evaluator = Evaluator(N_CLASSES)

    pbar = tqdm(data)
    for image, label in pbar:
        prediction = model(image)
        res = evaluator.add_sample(prediction.numpy(), label.numpy())
        pbar.set_description(f"Accuracy: {res['accuracy']}  mIOU: {res['miou']}")

    return evaluator.result()

def eval_model(model):
    data = create_cityscapes_ds("validation", 1)

    evaluator = Evaluator(N_CLASSES)

    pbar = tqdm(data)
    for image, label in pbar:
        prediction = model(image)
        res = evaluator.add_sample(prediction.numpy(), label.numpy())
        pbar.set_description(f"Accuracy: {res['accuracy']}  mIOU: {res['miou']}")

    return evaluator.result()


def eval_hls4ml_model(hls4ml_model_path):
    hls_model = hls4ml.converters.convert_from_config(hls4ml_model_path + '/hls4ml_config.yml')
    hls_model.compile()

    data = create_cityscapes_ds("validation", 1)

    evaluator = Evaluator(N_CLASSES)

    pbar = tqdm(data)
    for image, label in pbar:
        prediction = hls_model.predict(image.numpy()).reshape(1, HEIGHT, WIDTH, N_CLASSES)
        res = evaluator.add_sample(prediction, label.numpy())
        pbar.set_description(f"Accuracy: {res['accuracy']}  mIOU: {res['miou']}")

    return evaluator.result()

def eval_hls4ml_model_softmax(hls4ml_model_path):
    hls_model = hls4ml.converters.convert_from_config(hls4ml_model_path + '/hls4ml_config.yml')
    hls_model.compile()

    data = create_cityscapes_ds("validation", 1)

    evaluator = Evaluator(N_CLASSES)

    pbar = tqdm(data)
    for image, label in pbar:
        prediction = hls_model.predict(image.numpy()).reshape(1, HEIGHT, WIDTH, N_CLASSES)
        res = evaluator.add_sample(prediction, label.numpy())
        pbar.set_description(f"Accuracy: {res['accuracy']}  mIOU: {res['miou']}")

    return evaluator.result()

def get_evalers():
    return dict(
            keras=dict(func=eval_keras_model, help="Load keras model"),
            qkeras=dict(func=eval_qkeras_model, help="Load qkeras model"),
            hls4ml=dict(func=eval_hls4ml_model, help="Load hls4ml model"),
            hls4ml_softmax=dict(func=eval_hls4ml_model_softmax, help="Load hls4ml model with keras softmax")
    )


def parse_arguments():
    evalers = get_evalers()
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help="Types of models", dest="mode")
    for key, params in evalers.items():
        keras_parser = subparsers.add_parser(name=key, help=params["help"])
        keras_parser.add_argument("--model", "-m", type=str, help="Model path")
    return parser.parse_args()


def eval():
    args = parse_arguments()
    evalers = get_evalers()
    if args.mode in evalers:
        results = evalers[args.mode]["func"](args.model)
        print(json.dumps(results, indent=2))
    else:
        raise Exception(f"Mode {args.mode} not supported.")


if __name__ == "__main__":
    eval()
