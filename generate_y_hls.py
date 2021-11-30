import hls4ml
import numpy as np
hls4ml_model_path = '/eos/home-n/nghielme/enet-results-run2/results_hls_f8_clk7_rf6_q4_ap_fixed_8-4_/hls_f8_clk7_rf6_q4_ap_fixed_8-4__FIFO_OPT'
hls_model = hls4ml.converters.convert_from_config(hls4ml_model_path + '/hls4ml_config.yml')
hls_model.compile()
X = np.load('X_test.npy')
y_hls = hls_model.predict(X)
np.save('y_hls.npy', y_hls)