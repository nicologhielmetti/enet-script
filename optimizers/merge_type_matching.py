from hls4ml.model.hls_layers import FixedPrecisionType
from hls4ml.model.optimizer import OptimizerPass


class MergeTypeMatching(OptimizerPass):
    def match(self, node):
        is_match = (node.__class__.__name__ == 'Concatenate' or node.__class__.__name__ == 'Merge')
        return is_match

    def transform(self, model, node):
        i0 = model.get_layer_output_variable(node.inputs[0])
        i1 = model.get_layer_output_variable(node.inputs[1])
        out_precision = FixedPrecisionType(
            max(i0.type.precision.fractional, i1.type.precision.fractional) + max(i0.type.precision.integer,
                                                                                  i1.type.precision.integer),
            max(i0.type.precision.integer, i1.type.precision.integer))
        node.get_output_variable().type.precision = out_precision
        return False