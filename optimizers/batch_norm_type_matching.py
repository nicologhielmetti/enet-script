import re
from copy import deepcopy

from hls4ml.model.optimizer import OptimizerPass


class BNTypeMatching(OptimizerPass):

    def match(self, node):
        return node.__class__.__name__ == 'BatchNormalization' and \
               any(out_node.__class__.__name__ != 'Resize' for name in node.outputs
                   for out_node in node.get_output_nodes(output_name=name))

    def transform(self, model, node):
        out_precision = deepcopy(node.get_input_node().get_output_variable().type.precision)
        # scale precision is ap_int<x>
        scale_precision = node.weights['scale'].type.precision
        bias_precision = node.weights['bias'].type.precision
        # scale
        out_precision.integer += scale_precision.integer
        out_precision.fractional = max(scale_precision.fractional, out_precision.fractional)
        # bias
        out_precision.integer = min((max(bias_precision.integer, out_precision.integer) + 1), 5)
        out_precision.fractional = max(out_precision.fractional, bias_precision.fractional)
        out_precision.width = out_precision.fractional + out_precision.integer
        node.get_output_variable().type.precision = out_precision
        return False