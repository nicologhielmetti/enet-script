from hls4ml.model.optimizer import OptimizerPass


class ResizeTypeMatching(OptimizerPass):
    def match(self, node):
        is_match = (node.__class__.__name__ == 'Resize' and
                    (node.get_input_node().__class__.__name__ == 'Activation' or
                     node.get_input_node().__class__.__name__ == 'Conv2DBatchnorm')
                    )
        return is_match

    def transform(self, model, node):
        inode = node.get_input_node()
        in_out_type = inode.get_output_variable().type.precision
        out_out_var = node.get_output_variable()
        out_out_var.type.precision = in_out_type
        return False