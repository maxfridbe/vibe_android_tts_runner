"""Export adaptations, checked against the original frozen decoder."""
import operator
import copy

import torch

RANDOM_OPS = {"aten.randn_like.default", "aten.rand.default"}


class NoiseRecorder(torch.fx.Interpreter):
    def __init__(self, module):
        super().__init__(module)
        self.noise = []

    def run_node(self, node):
        value = super().run_node(node)
        if node.op == "call_function" and str(node.target) in RANDOM_OPS:
            self.noise.append(value.clone())
        return value


def externalize_noise(module):
    """Move random tensors to explicit inputs for reproducible GPU parity tests."""
    reference = torch.fx.GraphModule(module, copy.deepcopy(module.graph))
    graph = module.graph
    first_non_input = next(n for n in graph.nodes if n.op != "placeholder")
    count = 0
    for node in list(graph.nodes):
        if node.op == "call_function" and str(node.target) in RANDOM_OPS:
            with graph.inserting_before(first_non_input):
                noise = graph.placeholder(f"noise_{count}")
            node.replace_all_uses_with(noise)
            graph.erase_node(node)
            count += 1
    graph.eliminate_dead_code()
    graph.lint()
    module.recompile()
    return reference, count


def restore_shape_scalars(module):
    """Undo JIT's size -> scalar tensor -> arithmetic -> item round trips.

    Only expressions derived from tensor dimensions or literal constants are
    eligible. Values of input tensors are never specialized to sample data.
    """
    graph = module.graph
    scalars = {}
    replaced = 0
    binary = {"aten.add.Tensor": operator.add, "aten.sub.Tensor": operator.sub,
              "aten.mul.Tensor": operator.mul, "aten.div.Tensor": operator.truediv,
              "aten.floor_divide.default": operator.floordiv}

    def scalar(value):
        if isinstance(value, (int, float, bool)):
            return value
        return scalars.get(value)

    for node in list(graph.nodes):
        name = str(node.target)
        with graph.inserting_before(node):
            if node.op == "get_attr":
                value = module
                for part in node.target.split("."):
                    value = getattr(value, part)
                if isinstance(value, torch.Tensor) and value.numel() == 1:
                    scalars[node] = value.item()
            elif node.op == "call_function":
                if name in ("aten.sym_size.int", "aten.sym_size", "aten.size.int"):
                    scalars[node] = node
                elif name in ("aten.scalar_tensor", "aten.scalar_tensor.default"):
                    value = scalar(node.args[0])
                    if value is not None:
                        scalars[node] = value
                elif name in binary:
                    a, b = (scalar(v) for v in node.args[:2])
                    if a is not None and b is not None:
                        alpha = node.kwargs.get("alpha", 1)
                        if name in ("aten.add.Tensor", "aten.sub.Tensor") and alpha != 1:
                            b = graph.call_function(operator.mul, (b, alpha))
                        scalars[node] = graph.call_function(binary[name], (a, b))
                elif name in ("aten._to_copy.default", "aten.max.default"):
                    value = scalar(node.args[0])
                    if value is not None:
                        if node.kwargs.get("dtype") in (torch.int32, torch.int64):
                            value = graph.call_function(torch.sym_int, (value,))
                        scalars[node] = value
                elif name in ("aten._local_scalar_dense.default", "aten.item.default"):
                    value = scalar(node.args[0])
                    if value is not None:
                        if not isinstance(value, torch.fx.Node):
                            value = graph.call_function(operator.add, (value, 0))
                        node.replace_all_uses_with(value)
                        graph.erase_node(node)
                        replaced += 1
    graph.eliminate_dead_code()
    graph.lint()
    module.recompile()
    return replaced
