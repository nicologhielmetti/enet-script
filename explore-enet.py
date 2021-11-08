import tarfile

import hls4ml
import tensorflow

from hls4ml.model.profiling import optimize_fifos_depth
import argparse

from qkeras.utils import load_qmodel
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
import random as rnd

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


def get_model(n_filters=8, quantization=4):
    return load_qmodel(f'models_h5/hom{quantization}_32_{n_filters}_{n_filters}_{n_filters}_{n_filters}_{n_filters}.h5', compile=False)


def pack_results(dir):
    EXCLUDE_LIST = ['.autopilot', 'myproject_axi.wdb', 'xsim.dir']

    def exclude_function(tarinfo):
        if any(elem in tarinfo.name for elem in EXCLUDE_LIST):
            return None
        else:
            return tarinfo

    with tarfile.open('results_' + dir + '.tar.gz', mode='w:gz') as archive:
        archive.add(dir + '_FIFO_OPT', recursive=True, filter=exclude_function)


def get_dummy_model():
    rnd.seed(42)

    height = 10
    width = 10
    chan = 3

    input_shape = (height, width, chan)
    num_classes = 5

    model = Sequential()
    model.add(
        Conv2D(4, kernel_size=(3, 3), padding='same', strides=(1, 1), activation='linear', input_shape=input_shape,
               bias_initializer=tensorflow.keras.initializers.RandomUniform(minval=-0.8, maxval=0.8, seed=96),
               kernel_initializer=tensorflow.keras.initializers.RandomUniform(minval=-0.8, maxval=0.8, seed=96)))
    model.add(
        BatchNormalization(
            beta_initializer=tensorflow.keras.initializers.RandomUniform(minval=-0.8, maxval=0.8, seed=96),
            gamma_initializer=tensorflow.keras.initializers.RandomUniform(minval=-0.8, maxval=0.8, seed=96),
            moving_mean_initializer=tensorflow.keras.initializers.RandomUniform(minval=0, maxval=0.8,
                                                                                seed=96),
            moving_variance_initializer=tensorflow.keras.initializers.RandomUniform(minval=0, maxval=0.8,
                                                                                    seed=None)))
    model.add(ReLU(name='relu_1'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='mp2d_1'))
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax',
                    bias_initializer=tensorflow.keras.initializers.RandomUniform(minval=-0.8, maxval=0.8, seed=96),
                    kernel_initializer=tensorflow.keras.initializers.RandomUniform(minval=-0.8, maxval=0.8, seed=96)))
    return model


def get_dummy_model_and_build_hls(n_filters, clock_period, reuse_factor, quantization, precision='ap_fixed<8,4>',
                                  input_data=None, output_predictions=None):
    
    keras_model = get_dummy_model()
    
    hls_config = {
        'Model': {
            'Precision': precision,
            'ReuseFactor': reuse_factor,
            'Strategy': 'Resource',
            'FIFO_opt': 1,
        },
        'LayerName': {
            'dense_softmax': {
                'Strategy': 'Stable'
            }
        }
    }
    
    out_dir = 'hls_dummy_f{}_clk{}_rf{}_q{}_{}'.format(n_filters, clock_period, reuse_factor, quantization, precision) \
        .replace(',', '-').replace('<', '_').replace('>', '_')

    hls_model = optimize_fifos_depth(keras_model, output_dir=out_dir, clock_period=clock_period,
                                     backend='VivadoAccelerator',
                                     board='zcu102', hls_config=hls_config, input_data_tb=input_data,
                                     output_data_tb=output_predictions)
    hls4ml.templates.VivadoAcceleratorBackend.make_bitfile(hls_model)
    pack_results(out_dir)
    

def get_model_and_build_hls(n_filters, clock_period, reuse_factor, quantization, precision='ap_fixed<8,4>',
                            input_data=None,
                            output_predictions=None):
    keras_model = get_model(n_filters, quantization)

    hls_config = {
        'Model': {
            'Precision': precision,
            'ReuseFactor': reuse_factor,
            'Strategy': 'Resource',
            'FIFO_opt': 1,
        },
        'LayerName': {
            'conv2d_1': {
                'ConvImplementation': 'Encoded'
            }
        }
    }
    out_dir = 'hls_f{}_clk{}_rf{}_q{}_{}'.format(n_filters, clock_period, reuse_factor, quantization, precision) \
        .replace(',', '-').replace('<', '_').replace('>', '_')
    hls_model = optimize_fifos_depth(keras_model, output_dir=out_dir, clock_period=clock_period,
                                     backend='VivadoAccelerator',
                                     board='zcu102', hls_config=hls_config, input_data_tb=input_data,
                                     output_data_tb=output_predictions)
    hls4ml.templates.VivadoAcceleratorBackend.make_bitfile(hls_model)
    pack_results(out_dir)


parser = argparse.ArgumentParser()
parser.add_argument('-r', '--reuse_factor', type=int, help='Reuse factor', required=True)
parser.add_argument('-f', '--n_filters', type=int, help='Filter size', required=True)
parser.add_argument('-c', '--clock_period', type=int, help='HLS clock latency in ns', required=True)
parser.add_argument('-q', '--quantization', type=int, help='Uniform quantization of the model (i.e.: 4, 8)',
                    required=True)
parser.add_argument('-p', '--precision', type=str, help='Precision used by default in the hls model', nargs='?',
                    default='ap_fixed<8,4>')
parser.add_argument('-i', '--input_data', type=str, help='Input .npy file', nargs='?', default=None)
parser.add_argument('-o', '--output_predictions', type=str, help='Output .npy file', nargs='?', default=None)
args = parser.parse_args()

get_model_and_build_hls(n_filters=args.n_filters, clock_period=args.clock_period,
                        reuse_factor=args.reuse_factor, quantization=args.quantization, precision=args.precision,
                        input_data=args.input_data, output_predictions=args.output_predictions)
