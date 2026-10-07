"""FP32 replacements for Kitten's 16-point STFT/ISTFT, without complex tensors."""
import math
import torch
from torch.nn import functional as F


class RealSTFT16(torch.nn.Module):
    """ATen STFT input is already center-padded by the original traced graph."""
    def __init__(self, window):
        super().__init__()
        phase = 2 * math.pi * torch.arange(9, dtype=torch.float64)[:, None] * torch.arange(16, dtype=torch.float64) / 16
        basis = torch.cat((phase.cos(), -phase.sin())).float() * window
        self.register_buffer("basis", basis[:, None, :])

    def forward(self, x):
        spectrum = F.conv1d(x.unsqueeze(1), self.basis, stride=4)
        return spectrum.reshape(x.shape[0], 2, 9, -1).permute(0, 2, 3, 1)


class RealISTFT16(torch.nn.Module):
    def __init__(self, window):
        super().__init__()
        phase = 2 * math.pi * torch.arange(9, dtype=torch.float64)[:, None] * torch.arange(16, dtype=torch.float64) / 16
        scale = torch.full((9, 1), 2 / 16, dtype=torch.float64)
        scale[0] = scale[-1] = 1 / 16
        basis = torch.cat((phase.cos() * scale, -phase.sin() * scale)).float() * window
        self.register_buffer("basis", basis[:, None, None, :])
        self.register_buffer("window_squared", window.square()[None, None, None, :])

    def forward(self, real, imaginary):
        spectrum = torch.cat((real, imaginary), dim=1).unsqueeze(2)
        # Express overlap-add as 2D transposed convolution: the Vulkan backend
        # supports it, whereas its 1D transposed-convolution path is unsupported.
        signal = F.conv_transpose2d(spectrum, self.basis, stride=(1, 4))[:, 0, 0, 8:-8]
        envelope = F.conv_transpose2d(torch.ones_like(spectrum[:, :1]), self.window_squared,
                                      stride=(1, 4))[:, 0, 0, 8:-8]
        return signal / envelope


def transpose_conv_1d(x, weight, bias, stride, padding, dilation, transposed, output_padding, groups):
    assert transposed and len(stride) == 1
    return F.conv_transpose2d(x.unsqueeze(2), weight.unsqueeze(2), bias,
                             (1, stride[0]), (0, padding[0]), (0, output_padding[0]),
                             groups, (1, dilation[0])).squeeze(2)


def rewrite_audio_ops(module):
    graph = module.graph
    counts = {"stft": 0, "istft": 0, "transpose_conv_1d": 0}
    for node in list(graph.nodes):
        if node.op != "call_function":
            continue
        name = str(node.target)
        if name == "aten.view_as_real.default":
            source = node.args[0]
            if str(source.target) != "aten.stft.default":
                continue
            x, nfft, hop, win, window, normalized, onesided, complex_output, align = source.args
            assert (nfft, hop, win, normalized, onesided, complex_output, align) == (16, 4, 16, 0, None, 1, None)
            assert window.op == "get_attr" and len(source.users) == 1
            key = f"real_stft_{counts['stft']}"
            module.add_submodule(key, RealSTFT16(getattr(module, window.target)))
            with graph.inserting_before(node):
                replacement = graph.call_module(key, (x,))
            node.replace_all_uses_with(replacement)
            graph.erase_node(node)
            graph.erase_node(source)
            counts["stft"] += 1
        elif name == "aten.istft.default":
            source, nfft, hop, win, window, center, normalized, onesided, length, complex_output = node.args
            assert (nfft, hop, win, center, normalized, onesided, length, complex_output) == (16, 4, 16, 1, 0, None, None, 0)
            assert str(source.target) == "aten.complex.default" and len(source.users) == 1
            assert window.op == "get_attr"
            key = f"real_istft_{counts['istft']}"
            module.add_submodule(key, RealISTFT16(getattr(module, window.target)))
            with graph.inserting_before(node):
                replacement = graph.call_module(key, source.args)
            node.replace_all_uses_with(replacement)
            graph.erase_node(node)
            graph.erase_node(source)
            counts["istft"] += 1
        elif name == "aten.convolution.default" and node.args[6] and len(node.args[3]) == 1:
            node.target = transpose_conv_1d
            counts["transpose_conv_1d"] += 1
    graph.eliminate_dead_code()
    graph.lint()
    module.recompile()
    return counts
