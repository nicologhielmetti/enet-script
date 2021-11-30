import pandas as pd
from hls4ml.model import HLSModel
from hls4ml.model.profiling import numerical
from hls4ml.utils import config_from_keras_model
import hls4ml
from hls4ml.converters import convert_from_keras_model
import numpy as np
import qkeras
import matplotlib.pyplot as plt

from hls4ml.model.optimizer import OptimizerPass


class TypeMatching(OptimizerPass):
    def match(self, node):
        is_match = (node.__class__.__name__ == 'Resize' and
                    node.get_input_node().__class__.__name__ == 'Activation') or \
                   (node.__class__.__name__ == 'Concat' and node.get_input_node() is not None)
        return is_match

    def transform(self, model, node):
        inode = node.get_input_node()
        in_out_type = inode.get_output_variable().type.precision
        out_out_var = node.get_output_variable()
        out_out_var.type.precision = in_out_type
        return False


hls4ml.model.optimizer.register_pass('type_matching', TypeMatching)

print_plots = False
print_numerical = True
print_numerical_x = False

hls_path = 'results_run1/results_hls_f8_clk7_rf10_q8_ap_fixed_16-6_/hls_f8_clk7_rf10_q8_ap_fixed_16-6__FIFO_OPT'
keras_path = 'models_h5/hom8_32_8_8_8_8_8.h5'
X = np.load('X_test.npy')

keras_model = qkeras.utils.load_qmodel(keras_path, compile=False)

act_blocks_ymodel = np.load('session_savings/act_blocks_ymodel.npy', allow_pickle=True).tolist()
ymodel = {}
for d in act_blocks_ymodel:
    for k, v in d.items():
        ymodel[k] = v

hls4ml.model.optimizer.OutputRoundingSaturationMode.layers = ['Activation']
hls4ml.model.optimizer.OutputRoundingSaturationMode.rounding_mode = 'AP_RND'
hls4ml.model.optimizer.OutputRoundingSaturationMode.saturation_mode = 'AP_SAT'

config_def = config_from_keras_model(keras_model, granularity='name',
                                     default_precision='ap_fixed<16,6>',
                                     default_reuse_factor=10)
for layer in config_def['LayerName'].keys():
    config_def['LayerName'][layer]['Trace'] = True
config_def['LayerName']['input_1']['Precision'] = {'result': 'ap_ufixed<10,1>'}
config_def['LayerName']['conv_initial']['Precision'] = {
    'weight': 'ap_fixed<10,2>',
    'bias': 'ap_fixed<16,6>',
    'result': 'ap_fixed<16,4>',
    'default': 'ap_fixed<16,6>',
    'accum': 'ap_fixed<21,4>'
}
# config_def['LayerName']['batch_normalization']['Precision'] = {
#                                                                 'accum': 'ap_fixed<20,6>',
#                                                                 'scale': 'ap_fixed<16,6>',
#                                                                 'bias': 'ap_fixed<16,6>'
#                                                               }

hls_model = convert_from_keras_model(keras_model,
                                     output_dir='hls_f8_clk7_rf10_q8_ap_fixed_16-6__FIFO_OPT',
                                     backend='VivadoAccelerator', io_type='io_stream',
                                     board='zcu102', clock_period=7, hls_config=config_def)

hls_model.write()
_, ysim = hls_model.trace(X)

act_blocks_ysim = []

single_block_ysim = {}
for k, v in ysim.items():
    single_block_ysim[k] = v
    if 're_lu' in k:
        act_blocks_ysim.append(single_block_ysim)
        single_block_ysim = {}

df = pd.DataFrame()
block = 0
start = 0 if block == 0 else sum([len(e) for e in act_blocks_ysim[:block]])
stop = sum([len(e) for e in act_blocks_ysim[:block + 1]])
for layer in act_blocks_ysim[block].keys():
    df[layer + '_csim'] = pd.Series(ysim[layer].flatten())
    df[layer + '_qkeras'] = pd.Series(ymodel[layer].flatten())
    print('Max difference for layer       {}: {}'.format(layer,
                                                         np.abs(ysim[layer].flatten() - ymodel[layer].flatten()).max()))
    print('min/MAX values range for layer {}: {} - {}'.format(layer, ymodel[layer].flatten().min(),
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
