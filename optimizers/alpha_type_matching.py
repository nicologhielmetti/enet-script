from copy import deepcopy

from hls4ml.model.hls_layers import FixedPrecisionType
from hls4ml.model.optimizer import OptimizerPass


class AlphaTypeMatching(OptimizerPass):
    def match(self, node):
        is_match = (node.__class__.__name__ == 'ApplyAlpha')
        return is_match

    def transform(self, model, node):
        out_precision = deepcopy(node.get_input_node().get_output_variable().type.precision)
        # scale precision is ap_int<x>
        scale_width = node.weights['scale'].type.precision.width
        bias_precision = node.weights['bias'].type.precision
        scale_precision_ap_fixed = FixedPrecisionType(width=2*scale_width,
                                                      integer=scale_width)
        # scale
        out_precision.integer += scale_precision_ap_fixed.integer
        out_precision.fractional = max(scale_precision_ap_fixed.fractional, out_precision.fractional)
        # bias
        out_precision.integer = max(bias_precision.integer, out_precision.integer) + 1
        out_precision.fractional = max(out_precision.fractional, bias_precision.fractional)
        out_precision.width = out_precision.fractional + out_precision.integer
        node.get_output_variable().type.precision = out_precision
        return False