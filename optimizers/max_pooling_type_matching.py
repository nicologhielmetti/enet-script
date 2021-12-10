import re

from hls4ml.model.optimizer import OptimizerPass


class MP2DTypeMatching(OptimizerPass):

    def match(self, node):
        return node.__class__.__name__ == 'Pooling2D'

    def transform(self, model, node):
        in_name = 'clone_' + re.sub(r'_cpy\d+', '', node.inputs[0]) if 'cpy' in node.inputs[0] else None
        inode = node.get_input_node(input_name=in_name)
        in_out_type = inode.get_output_variable().type.precision
        out_out_var = node.get_output_variable()
        out_out_var.type.precision = in_out_type
        return False