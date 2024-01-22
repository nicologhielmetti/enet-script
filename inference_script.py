import numpy as np
from qkeras.utils import load_qmodel
import cv2
import matplotlib.pyplot as plt
import plotting


def plot_segmented_image(model_ouputs, image_to_plot):
    model_ouputs_indices = np.argmax(model_ouputs, axis=-1)
    model_ouputs = model_ouputs.astype(np.uint8)

    for x in range(model_ouputs_indices.shape[1]):
        for y in range(model_ouputs_indices.shape[2]):
            index = model_ouputs_indices[image_to_plot, x, y]

            if index == 0:
                model_ouputs[image_to_plot, x, y, 0] = 255
            elif index == 1:
                model_ouputs[image_to_plot, x, y, 0] = 85
            elif index == 2:
                model_ouputs[image_to_plot, x, y, 0] = 170
            else:
                model_ouputs[image_to_plot, x, y, 0] = 0

    plt.imshow(model_ouputs[image_to_plot, :, :, 0], cmap="hot")
    plt.colorbar()
    plt.show()

X = np.load('./data_pickles/X_test.npy')
# keras_model = load_qmodel('hls4ml_example_output/keras_model.h5', compile=False)
# y_keras = keras_model(X)

# plot_segmented_image(np.array(y_keras), 2)

import model_under_test

hls_model, keras_model, _ = model_under_test.get_hls_and_keras_models('hls4ml_example_output/keras_model.h5', 'ap_fixed<8,4>', 5, 7, 'model_3', False)

#y_qkeras = keras_model.predict(X)
#y_hls = hls_model.predict(X)
#np.save('model_3/y_qkeras.npy', y_qkeras)
#np.save('model_3/y_hls.npy', y_hls)
