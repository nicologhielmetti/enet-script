from hls4ml.model.optimizer import OptimizerPass


class CloneTypeMatching(OptimizerPass):

    def match(self, node):
        return node.__class__.__name__ == 'Clone'

    def transform(self, model, node):
        inode = node.get_input_node()
        in_out_type = inode.get_output_variable().type.precision
        for out_out_var in node.variables.values():
            out_out_var.type.precision = in_out_type
        return False
