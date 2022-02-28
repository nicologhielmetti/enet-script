import hls4ml
import qkeras
from hls4ml.converters import convert_from_keras_model
from hls4ml.utils import config_from_keras_model

from clone import CloneOutput
from optimizers.alpha_type_matching import AlphaTypeMatching
from optimizers.clone_type_matching import CloneTypeMatching
from optimizers.conv_type_matching import ConvTypeMatching
from optimizers.eliminate_linear import EliminateLinearActivation
from optimizers.eliminate_softmax import EliminateSoftmax
from optimizers.max_pooling_type_matching import MP2DTypeMatching
from optimizers.merge_type_matching import MergeTypeMatching
from optimizers.resize_type_matching import ResizeTypeMatching
from optimizers.zero_padding_type_matching import ZP2DTypeMatching


def get_hls_and_keras_models(model_path, default_precision, default_reuse_factor, clock_period, output_dir, trace):
    hls4ml.model.optimizer.OutputRoundingSaturationMode.layers = ['Activation', 'BatchNormalization']
    hls4ml.model.optimizer.OutputRoundingSaturationMode.rounding_mode = 'AP_RND_CONV'
    hls4ml.model.optimizer.OutputRoundingSaturationMode.saturation_mode = 'AP_SAT'

    dedicated_opt = {
        'eliminate_softmax': EliminateSoftmax,
        'eliminate_linear_v2': EliminateLinearActivation,
        'clone_output': CloneOutput,
        'conv_type_matching': ConvTypeMatching,
        'resize_type_matching': ResizeTypeMatching,
        'max_pooling_type_matching': MP2DTypeMatching,
        'clone_type_matching': CloneTypeMatching,
        'zero_padding_type_matching': ZP2DTypeMatching,
        'alpha_type_matching': AlphaTypeMatching,
        'merge_type_matching': MergeTypeMatching
    }

    for k, v in dedicated_opt.items():
        if k not in hls4ml.model.optimizer.get_available_passes():
            hls4ml.model.optimizer.register_pass(k, v)

    keras_model = qkeras.utils.load_qmodel(model_path, compile=False)
    config_def = config_from_keras_model(keras_model, granularity='name',
                                         default_precision=default_precision,
                                         default_reuse_factor=default_reuse_factor)

    config_def['LayerName']['input_1']['Precision'] = {
        'result': 'ap_ufixed<8,0>'
    }
    config_def['Model']['FIFO_opt'] = True
    config_def['Model']['Strategy'] = 'Resource'

    hls_model, config = get_hls_model(keras_model, config_def, clock_period, output_dir, trace)

    return hls_model, keras_model, config


def get_hls_model(keras_model, config, clock_period, output_dir, trace):
    hls_model = convert_from_keras_model(keras_model,
                                         output_dir=output_dir,
                                         backend='VivadoAccelerator', io_type='io_stream',
                                         board='zcu102', clock_period=clock_period, hls_config=config)
    if trace:
        for layer in list(hls_model.get_layers()):
            if layer.__class__.__name__ in ['ApplyAlpha', 'Clone']:
                continue
            try:
                config['LayerName'][layer.name]['Trace'] = True
            except KeyError:
                config['LayerName'][layer.name] = {'Trace': True}

        hls_model = convert_from_keras_model(keras_model,
                                             output_dir=output_dir,
                                             backend='VivadoAccelerator', io_type='io_stream',
                                             board='zcu102', clock_period=clock_period, hls_config=config)
    hls_model.compile()
    return hls_model, config
