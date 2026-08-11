#!/usr/bin/env python3
"""Export the PCA style basis as a flat binary the phone can read.

The on-device refine (CMA search over coefficients) needs to decode
coefficients back to style tensors and to encode a starting style into
coefficients — the two matmuls StyleBasis does. The shipped
style_encoder.onnx folds the basis into a graph that starts at an embedding,
so it cannot be used for a bare coeff<->style round trip; this dumps the basis
itself.

Little-endian layout, matching RefineEngine.kt's reader:
    int32  k, dTot, split, ttl_r, ttl_c, dp_r, dp_c
    float32 scale[k]
    float32 mean[dTot]
    float32 basis[k * dTot]      (row-major, k rows of dTot)

decode(c)  = mean + (c * scale) @ basis           -> split into ttl/dp, renorm
encode(s)  = ((flatten(s) - mean) @ basis.T) / scale

    export_basis.py --head trans_ecapa_u.pt --out ../models/cloner/style_basis.bin
"""
import argparse
import struct

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ck = torch.load(args.head, map_location="cpu", weights_only=False)
    basis = ck["basis"].numpy().astype(np.float32)      # (k, D)
    mean = ck["mean"].numpy().astype(np.float32)        # (D,)
    scale = ck["scale"].numpy().astype(np.float32)      # (k,)
    ttl_r, ttl_c = (int(v) for v in ck["ttl_shape"])
    dp_r, dp_c = (int(v) for v in ck["dp_shape"])
    k, dTot = basis.shape
    split = ttl_r * ttl_c
    assert split + dp_r * dp_c == dTot, "basis width must equal ttl+dp"

    with open(args.out, "wb") as f:
        f.write(struct.pack("<7i", k, dTot, split, ttl_r, ttl_c, dp_r, dp_c))
        f.write(scale.tobytes())
        f.write(mean.tobytes())
        f.write(basis.tobytes())
    print(f"wrote {args.out}: k={k} D={dTot} split={split} "
          f"ttl={ttl_r}x{ttl_c} dp={dp_r}x{dp_c} "
          f"({8 + (k + dTot + k * dTot) * 4} bytes)")

    # self-check: a coeff round trip through decode/encode returns the coeffs
    rng = np.random.default_rng(0)
    c = rng.standard_normal(k).astype(np.float32)
    flat = mean + (c * scale) @ basis
    back = ((flat - mean) @ basis.T) / scale
    print(f"encode(decode(c)) max abs err {np.abs(back - c).max():.2e} "
          "(basis rows orthonormal -> should be ~0)")


if __name__ == "__main__":
    main()
