import re

import numpy as np
from hls4ml.model.hls_layers import FixedPrecisionType
from hls4ml.model.optimizer import OptimizerPass


class ConvTypeMatching(OptimizerPass):

    def match(self, node):
        return node.__class__.__name__ == 'Conv2D' or node.__class__.__name__ == 'PointwiseConv2D' or node.__class__.__name__ == 'Conv2DBatchnorm'

    def transform(self, model, node):
        in_name = 'clone_' + re.sub(r'_cpy\d+', '', node.inputs[0]) if 'cpy' in node.inputs[0] else None
        input_bits = node.get_input_node(input_name=in_name).get_output_variable().type.precision.width
        input_integers = node.get_input_node(input_name=in_name).get_output_variable().type.precision.integer
        weight_bits = node.attributes['weight_quantizer'].bits
        weight_integers = node.attributes['weight_quantizer'].hls_type.integer
        if node.get_attr('bias_quantizer') == None:
            bias_bits = 0
            bias_integers = 0
        else:
            bias_bits = node.get_attr('bias_quantizer').bits
            bias_integers = node.get_attr('bias_quantizer').hls_type.integer
        n_ops = node.get_attr('n_chan') * node.get_attr('filt_height') * node.get_attr('filt_width')
        new_type = FixedPrecisionType(width=int(max(
                                                        np.ceil(input_bits + weight_bits + np.log2(n_ops)),
                                                        bias_bits
                                                    ) + 1),
                                      integer=int(max(
                                                          np.ceil(input_integers + weight_integers + np.log2(n_ops)),
                                                          bias_integers
                                                      ) + 1))
        node.set_attr('accum_t', new_type)
        node.get_output_variable().type.precision = new_type
        return False
