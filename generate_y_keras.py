from qkeras.utils import load_qmodel
import os
import numpy as np

X = np.load('X_test.npy')
for model in os.listdir('models_h5'):
    if model.endswith('.h5'):
        keras_model = load_qmodel('models_h5/' + model, compile=False)
        y_keras = keras_model(X)
        np.save('models_h5/y_keras_' + model[:8] + '.npy')
    else:
        continue