from hls4ml.converters import convert_from_config, parse_yaml_config
from hls4ml.model.profiling import numerical
import numpy as np
from qkeras.utils import load_qmodel
import matplotlib.pyplot as plt

hls_path = 'results_run1/results_hls_f8_clk7_rf10_q8_ap_fixed_16-6_/hls_f8_clk7_rf10_q8_ap_fixed_16-6__FIFO_OPT'
keras_path = 'models_h5/hom8_32_8_8_8_8_8.h5'
X = np.load('X_test.npy')
config = parse_yaml_config(hls_path + '/hls4ml_config.yml')
half_model = len(config['HLSConfig']['LayerName'].keys()) / 3
i = 0
for layer in config['HLSConfig']['LayerName'].keys():
    if i < half_model:
        config['HLSConfig']['LayerName'][layer]['Trace'] = True
        i += 1
    else:
        break
hls_model = convert_from_config(config)
# hls_model.compile()
keras_model = load_qmodel(keras_path, compile=False)
wp, wph, ap, aph = numerical(model=keras_model, hls_model=hls_model, X=X)
plt.show()
# y_hls = hls_model.predict(X)
# np.save(hls_path + '/y_hls.npy', y_hls)
