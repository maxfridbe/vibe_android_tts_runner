import unittest
import torch
from audio_ops import RealSTFT16, RealISTFT16, transpose_conv_1d


class AudioOperationsTest(unittest.TestCase):
    def test_real_fourier_ops_match_pytorch(self):
        torch.manual_seed(123)
        window = torch.hann_window(16)
        forward = RealSTFT16(window)
        inverse = RealISTFT16(window)
        for samples in (32, 508, 4096):
            with self.subTest(samples=samples):
                x = torch.randn(1, samples)
                padded = torch.nn.functional.pad(x.unsqueeze(1), (8, 8), mode="reflect").squeeze(1)
                expected = torch.stft(x, 16, 4, 16, window, return_complex=True)
                actual = forward(padded)
                torch.testing.assert_close(actual, torch.view_as_real(expected), atol=2e-6, rtol=2e-6)
                audio = inverse(actual[..., 0], actual[..., 1])
                reference = torch.istft(expected, 16, 4, 16, window)
                torch.testing.assert_close(audio, reference, atol=2e-6, rtol=2e-6)

    def test_2d_transpose_convolution_preserves_1d_results(self):
        torch.manual_seed(45)
        for stride, groups in ((3, 1), (5, 1), (2, 2)):
            with self.subTest(stride=stride, groups=groups):
                x = torch.randn(1, 8, 19)
                weight = torch.randn(8, 4, 2 * stride)
                bias = torch.randn(4 * groups)
                expected = torch.nn.functional.conv_transpose1d(x, weight, bias,
                    stride=stride, padding=stride // 2, groups=groups)
                actual = transpose_conv_1d(x, weight, bias, [stride], [stride // 2],
                                           [1], True, [0], groups)
                torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
