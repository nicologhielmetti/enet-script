import re

import numpy as np
from hls4ml.model.hls_layers import FixedPrecisionType
from hls4ml.model.optimizer import OptimizerPass


class ConvTypeMatching(OptimizerPass):

    def match(self, node):
        return node.__class__.__name__ == 'Conv2D' or node.__class__.__name__ == 'PointwiseConv2D'

    def transform(self, model, node):
        in_name = 'clone_' + re.sub(r'_cpy\d+', '', node.inputs[0]) if 'cpy' in node.inputs[0] else None
        input_bits = node.get_input_node(input_name=in_name).get_output_variable().type.precision.width
        input_integers = node.get_input_node(input_name=in_name).get_output_variable().type.precision.integer
        weight_bits = node.get_attr('weight_quantizer').bits
        weight_integers = node.get_attr('weight_quantizer').hls_type.integer
        n_ops = node.get_attr('n_chan') * node.get_attr('filt_height') * node.get_attr('filt_width')
        new_type = FixedPrecisionType(width=int(np.ceil(input_bits + weight_bits + np.log2(n_ops))),
                                      integer=int(np.ceil(input_integers + weight_integers + np.log2(n_ops))))
        node.set_attr('accum_t', new_type)
        node.get_output_variable().type.precision = new_type
        return False
