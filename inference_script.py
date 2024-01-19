# %%
import numpy as np
import cv2
from qkeras.utils import load_qmodel
import hls4ml
import model_under_test

# %%
X = np.load("data_pickles/X_test.npy")

# %%
from matplotlib import pyplot as plt

image = cv2.cvtColor(X[2, :, :, :], cv2.COLOR_BGR2RGB)
plt.imshow(image)
plt.show()


# %%
def produce_hls_and_keras_model_outputs(model_inputs, path_to_keras_model):
    # hls_model = hls4ml.converters.convert_from_config(
    #     path_to_hls4ml_config_yml + "/hls4ml_config.yml"
    # )
    hls_model, keras_model, _ = model_under_test.get_hls_and_keras_models(path_to_keras_model, "ap_fixed<8,4>", 8, 7, "hls4ml_example_output0", False)

    # hls_model.compile()
    return hls_model.predict(model_inputs), keras_,model(model_inputs)


# def produce_keras_model_output(model_inputs, path_to_keras_model, keras_model_name):
#     keras_model = load_qmodel(
#         path_to_keras_model + "/" + keras_model_name, compile=False
#     )
#     return keras_model(model_inputs)


# %%
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


# %%
## Keras Model Inference

y_hls, y_keras = produce_hls_and_keras_model_outputs(X, "models_h5_run2", "hom4_32_16_16_16_16_16.h5")

# %%
# plot_segmented_image(np.array(y_keras), 2)

# %%
## hls4ml Model Inference

# %%
