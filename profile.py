import argparse

import pandas as pd
from hls4ml.model.profiling import numerical, get_ymodel_keras
import numpy as np
import matplotlib.pyplot as plt

from model_under_test import get_hls_and_keras_models

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

block = 38
print_plots = True
print_numerical = False
print_numerical_x = False
print_precision = True
create_y_model = False

ymodel_path = 'session_savings/y_model2_hom{}_32_{}_{}_{}_{}_{}.npy'\
    .format(args.quantization, args.n_filters, args.n_filters, args.n_filters, args.n_filters, args.n_filters)

X = np.load(args.input_data)

model_path = 'models_h5_run2/hom{}_32_{}_{}_{}_{}_{}.h5'\
        .format(args.quantization, args.n_filters, args.n_filters, args.n_filters, args.n_filters, args.n_filters)

out_dir = 'hls_f{}_clk{}_rf{}_q{}_{}_test'.format(args.n_filters, args.clock_period, args.reuse_factor, args.quantization,
                                             args.precision).replace(',', '-').replace('<', '_').replace('>', '_')

hls_model, keras_model, config = get_hls_and_keras_models(model_path, args.precision, args.reuse_factor,
                                                          args.clock_period, out_dir, True)

if create_y_model:
    ymodel = get_ymodel_keras(keras_model=keras_model, X=X)
    single_block_ymodel = {}
    act_blocks_ymodel = []
    for k, v in ymodel.items():
        single_block_ymodel[k] = v
        if 're_lu' in k:
            act_blocks_ymodel.append(single_block_ymodel)
            single_block_ymodel = {}
        else:
            act_blocks_ymodel.append(single_block_ymodel)
            single_block_ymodel = {}
    np.save(ymodel_path, act_blocks_ymodel)

act_blocks_ymodel = np.load(ymodel_path, allow_pickle=True).tolist()

ymodel = {}
for d in act_blocks_ymodel:
    for k, v in d.items():
        ymodel[k] = v

hls_model.write()
_, ysim = hls_model.trace(X)

act_blocks_ysim = []
single_block_ysim = {}
for k, v in ysim.items():
    single_block_ysim[k] = v
    if 're_lu' in k:
        act_blocks_ysim.append(single_block_ysim)
        single_block_ysim = {}
    else:
        act_blocks_ymodel.append(single_block_ysim)
        single_block_ysim = {}

df = pd.DataFrame()
start = 0 if block == 0 else sum([len(e) for e in act_blocks_ysim[:block]])
stop = sum([len(e) for e in act_blocks_ysim[:block + 1]])
for layer in act_blocks_ysim[block].keys():
    df[layer + '_csim'] = pd.Series(ysim[layer].flatten())
    df[layer + '_qkeras'] = pd.Series(ymodel[layer].flatten())
    print('Max difference for layer               {}: {}'.format(layer,
                                                                 np.abs(ysim[layer].flatten() - ymodel[layer].flatten()).max()))
    n_wrong = len([e for e in np.abs(ysim[layer].flatten() - ymodel[layer].flatten()) if e > 1e-4])
    print('Number of mismatching values for layer {}:  {}, {}%'.format(layer, n_wrong,
                                                                       100 * n_wrong / len(ysim[layer].flatten())))
    print('min/MAX values range for layer         {}: {} - {}'.format(layer, ymodel[layer].flatten().min(),
                                                              ymodel[layer].flatten().max()))
    if not print_plots:
        continue
    plt.figure()
    plt.scatter(ysim[layer].flatten(), ymodel[layer].flatten(), s=0.2)
    min_x = min(np.amin(ysim[layer]), np.amin(ymodel[layer]))
    max_x = max(np.amax(ysim[layer]), np.amax(ymodel[layer]))
    plt.plot([min_x, max_x], [min_x, max_x], c='gray')
    plt.xlabel('hls4ml {}'.format(layer))
    plt.ylabel('QKeras {}'.format(layer))
    plt.show()

if print_numerical:
    if print_numerical_x:
        wp, wph, ap, aph = numerical(model=keras_model, hls_model=hls_model, X=X, start=start, stop=stop)
    else:
        wp, wph, ap, aph = numerical(model=keras_model, hls_model=hls_model, start=start, stop=stop)
    plt.show()

if print_precision:
    print('--- PRECISION ---')
    block_counter = 0
    for l in list(hls_model.get_layers()):
        if block_counter == block:
            print('Layer: ' + l.name)
            for k, v in l.variables.items():
                print('- variable: ' + k + ' | precision: ' + str(v.type.precision))
        if 're_lu' in l.name:
            block_counter += 1
