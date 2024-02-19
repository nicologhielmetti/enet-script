import hls4ml
import numpy as np
hls4ml_model_path = './hls4ml_example_output'
hls_model = hls4ml.converters.convert_from_config(hls4ml_model_path + '/hls4ml_config.yml')
hls_model.compile()
X = np.load('./data_pickles/X_test.npy')
y_hls = hls_model.predict(X)
np.save('./data_pickles/y_hls.npy', y_hls)