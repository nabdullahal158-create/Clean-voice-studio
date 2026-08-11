#!/usr/bin/env python3
"""Docker build-এর সময়ে চলে: dns48 মডেলটা ১০ সেকেন্ডের ফিক্সড উইন্ডোতে ONNX-এ এক্সপোর্ট করে।
রানটাইমে torch লাগে না — শুধু ছোট onnxruntime দিয়ে চলে (~400MB RAM)।"""
import warnings
warnings.filterwarnings("ignore")

import torch  # noqa
from denoiser import pretrained  # noqa

OUT = "/out/dns48_5s.onnx"

m = pretrained.dns48().cpu().eval()
x = torch.randn(1, 1, 80000)  # 5s @ 16kHz — frozen window (ছোট উইন্ডো = কম RAM)
torch.onnx.export(
    m, x, OUT,
    input_names=["x"], output_names=["y"],
    dynamic_axes={"x": {2: "T"}, "y": {2: "T"}},
    opset_version=17, dynamo=False,
)
import os
print("✅ ONNX exported:", OUT, round(os.path.getsize(OUT) / 1e6, 1), "MB")
