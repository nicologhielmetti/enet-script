import pandas as pd
from hls4ml.model.profiling import numerical
import numpy as np
import matplotlib.pyplot as plt

from model_under_test import get_hls_and_keras_model

block = 0
print_plots = True
print_numerical = False
print_numerical_x = False
print_precision = True

hls_path = 'results_run1/results_hls_f8_clk7_rf10_q8_ap_fixed_16-6_/hls_f8_clk7_rf10_q8_ap_fixed_16-6__FIFO_OPT'
keras_path = 'models_h5/hom8_32_8_8_8_8_8.h5'
X = np.load('X_test_256.npy')

act_blocks_ymodel = np.load('session_savings/y_model_divided_256.npy', allow_pickle=True).tolist()
ymodel = {}
for d in act_blocks_ymodel:
    for k, v in d.items():
        ymodel[k] = v

hls_model, keras_model = get_hls_and_keras_model(keras_path, trace=True)

# ymodel = hls4ml.model.profiling.get_ymodel_keras(keras_model=keras_model, X=X)


# np.save('session_savings/y_model_divided_256.npy', act_blocks_ymodel)
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
