import torch
from torch.utils.cpp_extension import load_inline

cpp_code = ""
cuda_code = ""

module = load_inline(
    name="linear_attn_causal_ext",
    cpp_sources=cpp_code,
    cuda_sources=cuda_code,
    functions=["linear_attn_causal_forward_cuda", "linear_attn_causal_backward_cuda"],
    with_cuda=True,
    extra_cuda_cflags=["-O3", "-allow-unsupported-compiler"],
)
