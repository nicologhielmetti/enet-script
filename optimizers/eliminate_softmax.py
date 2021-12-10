from hls4ml.model.optimizer import OptimizerPass


class EliminateSoftmax(OptimizerPass):
    def match(self, node):
        if node.__class__.__name__ == 'Softmax':
            return True

    def transform(self, model, node):
        model.remove_node(node)
        return True