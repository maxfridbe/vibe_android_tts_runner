#!/usr/bin/env python3
"""Audit and export the pinned Kitten student decoder for Vulkan 1.1.

Development tool: writes artifacts outside the app until numerical parity and
device execution have passed. No model weights are embedded in this script.
"""
import argparse
import collections
import importlib.metadata
import json
from pathlib import Path
import time
import traceback

import torch
from graph_passes import restore_shape_scalars, externalize_noise, NoiseRecorder
from audio_ops import rewrite_audio_ops
from fetch_assets import checksum

COMPILE_OPTIONS = {"force_fp16": False, "texture_limits": (256, 256, 256),
                   "buffer_limit": 128 * 1024 * 1024}


def checkpoint(path, report):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(path)


def inputs(voices, length):
    ref = voices["Bruno"]
    return (
        torch.full((1, length), 4299, dtype=torch.int64),
        torch.tensor(ref["prompt_token"], dtype=torch.int64),
        torch.tensor(ref["prompt_feat"], dtype=torch.float32),
        torch.tensor(ref["embedding"], dtype=torch.float32),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decoder", type=Path, required=True)
    parser.add_argument("--voices", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--export-only", action="store_true",
                        help="Stop after CPU parity and torch.export; skip Vulkan lowering")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "target": {"vulkan": "1.1", "dtype": "float32", "device_validated": False,
                   "compile_options": COMPILE_OPTIONS},
        "decoder_sha256": checksum(args.decoder),
        "versions": {p: importlib.metadata.version(p) for p in ("torch", "executorch")},
        "status": "running",
        "static_export_tokens": 17,
    }
    torch.set_num_threads(4)
    stage = "load"
    try:
        module = torch.jit.load(str(args.decoder), map_location="cpu").eval()
        voices = json.loads(args.voices.read_text())
        report["torchscript_ops"] = dict(sorted(collections.Counter(
            n.kind() for n in module.inlined_graph.nodes()).items()))
        report["baseline"] = []
        stage = "cpu_baseline"
        with torch.inference_mode():
            for length in (4, 17, 53):
                torch.manual_seed(123)
                started = time.perf_counter()
                audio = module(*inputs(voices, length))
                assert torch.isfinite(audio).all() and audio.numel() > 0
                report["baseline"].append({"tokens": length, "samples": audio.numel(),
                    "seconds": time.perf_counter() - started})
        if args.audit_only:
            report["status"] = "audited"
            return
        stage = "torch_export"
        print("Converting frozen TorchScript to torch.export", flush=True)
        from torch._export.converter import TS2EPConverter
        class GraphConverter(TS2EPConverter):
            def retrace_as_exported_program(self, graph, constants):
                return graph

        graph = GraphConverter(module, inputs(voices, 17)).convert()
        report["shape_roundtrips_removed"] = restore_shape_scalars(graph)
        reference, report["noise_inputs"] = externalize_noise(graph)
        report["audio_rewrites"] = rewrite_audio_ops(graph)
        (args.output / "decoder_fx.py").write_text(graph.code)
        report["parity_cases"] = []
        with torch.inference_mode():
            for length in (4, 17, 53, 101):
                torch.manual_seed(123)
                recorder = NoiseRecorder(reference)
                expected = recorder.run(*inputs(voices, length))
                torch.manual_seed(123)
                original = module(*inputs(voices, length))
                torch.testing.assert_close(expected, original, rtol=2e-4, atol=2e-4)
                # The traced vocoder mutates its phase-noise input. Keep pristine
                # inputs for export and the native runner's bundled parity test.
                case_inputs = (*inputs(voices, length), *recorder.noise)
                actual = graph(*(tensor.clone() for tensor in case_inputs))
                torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-4)
                report["parity_cases"].append({"tokens": length, "samples": actual.numel(),
                    "max_absolute_error": (actual - expected).abs().max().item(),
                    "noise_shapes": [list(t.shape) for t in recorder.noise]})
                if length == 17:
                    sample = tuple(t.clone() for t in case_inputs)
                    sample_expected = expected.clone()
        report["fx_parity"] = True
        checkpoint(args.output / "report.json", report)
        exported = torch.export.export(graph, sample, strict=False)
        torch.export.save(exported, args.output / "decoder.pt2")
        torch.save({"inputs": sample, "expected": sample_expected}, args.output / "parity_case.pt")
        if args.export_only:
            report["status"] = "torch_exported_cpu_parity_passed"
            return
        stage = "vulkan_lowering"
        report["stage"] = stage
        checkpoint(args.output / "report.json", report)
        from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
        from executorch.exir import to_edge_transform_and_lower
        program = to_edge_transform_and_lower(exported, partitioner=[VulkanPartitioner(
            compile_options=COMPILE_OPTIONS)]).to_executorch()
        with (args.output / "decoder.pte").open("wb") as output:
            program.write_to_file(output)
        report["execution_plans"] = [{"name": plan.name,
            "delegates": dict(collections.Counter(d.id for d in plan.delegates)),
            "cpu_operators": [f"{op.name}.{op.overload}" for op in plan.operators]}
            for plan in program.executorch_program.execution_plan]
        # Official executor_runner accepts this bundle and checks real output,
        # instead of merely proving that shaders can be loaded.
        from executorch.devtools import BundledProgram
        from executorch.devtools.bundled_program.config import MethodTestCase, MethodTestSuite
        from executorch.devtools.bundled_program.serialize import serialize_from_bundled_program_to_flatbuffer
        bundle = BundledProgram(program, [MethodTestSuite("forward", [
            MethodTestCase(inputs=sample, expected_outputs=[sample_expected])])])
        (args.output / "decoder.bpte").write_bytes(serialize_from_bundled_program_to_flatbuffer(bundle))
        report["status"] = "exported_unvalidated"
    except Exception as error:
        report.update(status="blocked", stage=stage, error=f"{type(error).__name__}: {error}"[:8000])
        (args.output / "failure.txt").write_text(traceback.format_exc())
        raise
    finally:
        checkpoint(args.output / "report.json", report)
        print(json.dumps({k: v for k, v in report.items() if k != "torchscript_ops"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
