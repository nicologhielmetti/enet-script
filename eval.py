import argparse

import tensorflow as tf
import numpy as np
from tqdm import tqdm
import json
import qkeras
# import hls4ml

from cityscapes import create_cityscapes_ds, N_CLASSES, WIDTH, HEIGHT
from model_under_test import get_hls_and_keras_models

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


def eval_hls4ml_model(quantization, n_filters, default_precision, default_reuse_factor, clock_period):
    model_path = 'models_h5_run2/hom{}_32_{}_{}_{}_{}_{}.h5' \
        .format(quantization, n_filters, n_filters, n_filters, n_filters, n_filters)
    out_dir = 'hls_f{}_clk{}_rf{}_q{}_{}_PROFILE'.format(n_filters, clock_period, default_reuse_factor,
                                                         quantization,
                                                         default_precision).replace(',', '-') \
        .replace('<', '_').replace('>', '_')
    hls_model, _, _ = get_hls_and_keras_models(model_path, default_precision, default_reuse_factor, clock_period,
                                               out_dir, False)
    data = create_cityscapes_ds("validation", 1)

    evaluator = Evaluator(N_CLASSES)

    pbar = tqdm(data)
    for image, label in pbar:
        prediction = hls_model.predict(image.numpy()).reshape(1, HEIGHT, WIDTH, N_CLASSES)
        res = evaluator.add_sample(prediction, label.numpy())
        pbar.set_description(f"Accuracy: {res['accuracy']}  mIOU: {res['miou']}")

    return evaluator.result()


def eval_hls4ml_vs_qkeras(quantization, n_filters, default_precision, default_reuse_factor, clock_period):
    def evaluate(model, data):
        evaluator = Evaluator(N_CLASSES)
        if model.__class__.__name__ == "HLSModel":
            y_hls = []
            y_test = []
            X_test = []
        pbar = tqdm(data)
        for image, label in pbar:
            prediction = model.predict(image.numpy()).reshape(1, HEIGHT, WIDTH, N_CLASSES)
            if model.__class__.__name__ == "HLSModel":
                y_hls.append(prediction)
                y_test.append(label.numpy())
                X_test.append(image.numpy())
            res = evaluator.add_sample(prediction, label.numpy())
            pbar.set_description(f"Accuracy: {res['accuracy']}  mIOU: {res['miou']}")

        if model.__class__.__name__ == "HLSModel":
            np.save(out_dir + '/X_test.npy', X_test)
            np.save(out_dir + '/y_test.npy', y_test)
            np.save(out_dir + '/y_hls.npy', y_hls)

        return evaluator.result()

    model_path = 'models_h5_run2/hom{}_32_{}_{}_{}_{}_{}.h5' \
        .format(quantization, n_filters, n_filters, n_filters, n_filters, n_filters)
    out_dir = 'hls_f{}_clk{}_rf{}_q{}_{}_test_14_jan_FIFO_OPT'.format(n_filters, clock_period, default_reuse_factor,
                                                         quantization,
                                                         default_precision).replace(',', '-') \
        .replace('<', '_').replace('>', '_')
    hls_model, qkeras_model, _ = get_hls_and_keras_models(model_path, default_precision, default_reuse_factor,
                                                          clock_period, out_dir, False)
    data = create_cityscapes_ds("validation", 1)
    hls_result = evaluate(hls_model, data)
    qkeras_result = evaluate(qkeras_model, data)
    results = {'hls4ml': hls_result, 'qkeras': qkeras_result}
    return results


def get_evalers():
    return dict(
        keras=dict(func=eval_keras_model, help="Load keras model"),
        qkeras=dict(func=eval_qkeras_model, help="Load qkeras model"),
        hls4ml=dict(func=eval_hls4ml_model, help="Load hls4ml model"),
        hls4ml_vs_qkeras=dict(func=eval_hls4ml_vs_qkeras, help="Load hls4ml model and qkeras")
    )


def parse_arguments():
    evalers = get_evalers()
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help="Types of models", dest="mode")
    for key, params in evalers.items():
        keras_parser = subparsers.add_parser(name=key, help=params["help"])
        if key == 'hls4ml' or key == 'hls4ml_vs_qkeras':
            keras_parser.add_argument('-r', '--reuse_factor', type=int, help='Reuse factor', required=True)
            keras_parser.add_argument('-f', '--n_filters', type=int, help='Filter size', required=True)
            keras_parser.add_argument('-c', '--clock_period', type=int, help='HLS clock latency in ns', required=True)
            keras_parser.add_argument('-q', '--quantization', type=int,
                                      help='Uniform quantization of the model (i.e.: 4, 8)',
                                      required=True)
            keras_parser.add_argument('-p', '--precision', type=str, help='Precision used by default in the hls model',
                                      nargs='?',
                                      default='ap_fixed<8,4>')
        else:
            keras_parser.add_argument("--model", "-m", type=str, help="Model path")

    return parser.parse_args()


def eval():
    args = parse_arguments()
    evalers = get_evalers()
    if args.mode in evalers:
        if args.mode == 'hls4ml' or args.mode == 'hls4ml_vs_qkeras':
            results = evalers[args.mode]["func"](args.quantization, args.n_filters, args.precision, args.reuse_factor,
                                                 args.clock_period)
        else:
            results = evalers[args.mode]["func"](args.model)
        print(json.dumps(results, indent=2))
    else:
        raise Exception(f"Mode {args.mode} not supported.")


if __name__ == "__main__":
    eval()
