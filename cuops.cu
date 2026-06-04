#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cstdio>
#include <cmath>

// ═══════════════════════════════════════════════════════════════════════════════
// Device utilities
// ═══════════════════════════════════════════════════════════════════════════════

constexpr int WARP_SZ = 32;
constexpr float EPS  = 1e-8f;

template<typename T>
__device__ __forceinline__ T warp_reduce_sum(T val) {
    #pragma unroll
    for (int off = WARP_SZ / 2; off > 0; off /= 2)
        val += __shfl_down_sync(0xffffffff, val, off);
    return val;
}

// Block-level sum reduction.
// Returns the total sum in lane 0 of every warp (correct only in warp 0).
template<typename T>
__device__ __forceinline__ T block_reduce_sum(
    T val, T* scratch, int tid, int block_sz
) {
    int wid = tid / WARP_SZ;
    int lid = tid % WARP_SZ;
    int nw  = (block_sz + WARP_SZ - 1) / WARP_SZ;

    val = warp_reduce_sum(val);
    if (lid == 0) scratch[wid] = val;
    __syncthreads();

    val = (lid < nw) ? scratch[lid] : T(0);
    if (wid == 0) val = warp_reduce_sum(val);
    return val;
}


// ── Error-checking helper ────────────────────────────────────────────────────
#define CUDA_KERNEL_CHECK()                                         \
    do {                                                            \
        cudaError_t _err = cudaGetLastError();                      \
        if (_err != cudaSuccess) {                                  \
            fprintf(stderr, "CUDA error at %s:%d : %s\n",           \
                    __FILE__, __LINE__, cudaGetErrorString(_err));   \
        }                                                           \
    } while (0)


// ═══════════════════════════════════════════════════════════════════════════════
// Kernel 1 — linear_kernels_attn_causal  (fused, causal / cumsum)
// ═══════════════════════════════════════════════════════════════════════════════
//
// Grid:  (B,  d_v)
// Block: (min(D, 256),)
// Shared memory: 2 * num_warps * sizeof(float) for reduction scratch.
//
// Each block handles one (batch, dv) pair.  Threads tile over D.
// At each timestep l we:
//   1. load   φ_l, ψ_l, V_l   for our d-chunk
//   2. update running accumulators  C_r += ψ,  S_r += ψ · V
//   3. compute partial dot products  Σ φ·S_r,  Σ φ·C_r
//   4. block-reduce to get numerator & denominator
//   5. write  Y[b,l,dv] = num / max(den, ε)
// ---------------------------------------------------------------------------
template<typename scalar_t>
__global__ void linear_kernels_attn_causal_kernel(
    const scalar_t* __restrict__ phi,
    const scalar_t* __restrict__ psi,
    const scalar_t* __restrict__ V,
    scalar_t*       __restrict__ out,
    const int B, const int L, const int D, const int d_v
) {
    int b   = blockIdx.x;
    int dv  = blockIdx.y;
    int tid = threadIdx.x;
    int bsz = blockDim.x;                 // = min(D, 256)

    // Shared scratch for block reductions
    extern __shared__ float shm[];
    int nw = (bsz + WARP_SZ - 1) / WARP_SZ;
    float* num_scr = shm;
    float* den_scr = shm + nw;

    // Each thread owns a contiguous slice of D
    int d_per = (D + bsz - 1) / bsz;
    int d_beg = tid * d_per;
    int d_end = min(D, d_beg + d_per);
    int my_n  = max(0, d_end - d_beg);

    // Running accumulators (register file; spills to local mem if large)
    float* C_acc = (float*)alloca(my_n * sizeof(float));
    float* S_acc = (float*)alloca(my_n * sizeof(float));
    for (int k = 0; k < my_n; ++k) { C_acc[k] = 0.f; S_acc[k] = 0.f; }

    int phi_sb = L * D;        // stride between batches for phi
    int psi_sb = L * D;        //   "    for psi
    int V_sb   = L * d_v;      //   "    for V

    for (int l = 0; l < L; ++l) {
        float p_num = 0.f;
        float p_den = 0.f;

        for (int k = 0; k < my_n; ++k) {
            int d = d_beg + k;

            float phi_v = static_cast<float>(phi[b * phi_sb + l * D       + d]);
            float psi_v = static_cast<float>(psi[b * psi_sb + l * D       + d]);
            float V_v   = static_cast<float>(  V[b * V_sb   + l * d_v     + dv]);

            C_acc[k] += psi_v;
            S_acc[k] += psi_v * V_v;

            p_num += phi_v * S_acc[k];
            p_den += phi_v * C_acc[k];
        }

        p_num = block_reduce_sum(p_num, num_scr, tid, bsz);
        p_den = block_reduce_sum(p_den, den_scr, tid, bsz);

        if (tid == 0) {
            float d_clamped = fmaxf(p_den, EPS);
            out[b * (L * d_v) + l * d_v + dv] =
                static_cast<scalar_t>(p_num / d_clamped);
        }
        __syncthreads();
    }
}


// ═══════════════════════════════════════════════════════════════════════════════
// Kernel 2a — compute_S  (global reduction: S[b,d,dv] = Σ_l ψ·V)
// ═══════════════════════════════════════════════════════════════════════════════
//
// Simple per-(b,d) parallel reduction over L, producing all d_v entries.
// Block: (min(d_v, 256),)   — one thread per output dv value.
// Grid:  (B, D)
// Each thread walks the entire sequence L.
// ---------------------------------------------------------------------------
template<typename scalar_t>
__global__ void compute_S_kernel(
    const scalar_t* __restrict__ psi,   // (B, L, D)
    const scalar_t* __restrict__ V,     // (B, L, d_v)
    float*          __restrict__ S_out, // (B, D, d_v)
    const int B, const int L, const int D, const int d_v
) {
    int b  = blockIdx.x;
    int d  = blockIdx.y;
    int dv = threadIdx.x;
    if (dv >= d_v) return;

    int psi_sb = L * D;
    int V_sb   = L * d_v;

    float acc = 0.f;
    for (int l = 0; l < L; ++l) {
        float psi_v = static_cast<float>(psi[b * psi_sb + l * D   + d]);
        float V_v   = static_cast<float>(V  [b * V_sb   + l * d_v + dv]);
        acc += psi_v * V_v;
    }
    S_out[(b * D + d) * d_v + dv] = acc;
}


// ═══════════════════════════════════════════════════════════════════════════════
// Kernel 2b — compute_output  (non-causal output from precomputed C, S)
// ═══════════════════════════════════════════════════════════════════════════════
//
// Grid:  (B * L,  ceil(d_v / BLK_DV))
// Block: (min(d_v, BLK_DV),)
// ---------------------------------------------------------------------------
template<typename scalar_t>
__global__ void linear_kernels_attn_output_kernel(
    const scalar_t* __restrict__ phi,    // (B, L, D)
    const float*    __restrict__ C,      // (B, D)
    const float*    __restrict__ S,      // (B, D, d_v)
    scalar_t*       __restrict__ out,    // (B, L, d_v)
    const int B, const int L, const int D, const int d_v
) {
    int bl       = blockIdx.x;
    int b        = bl / L;
    int l        = bl % L;
    int dv_tile  = blockIdx.y;

    constexpr int BLK_DV = 64;
    int dv_start = dv_tile * BLK_DV;
    int dv       = dv_start + threadIdx.x;
    if (dv >= d_v) return;

    float num = 0.f, den = 0.f;

    int phi_off = b * (L * D) + l * D;
    int C_off   = b * D;
    int S_off   = (b * D) * d_v;

    for (int d = 0; d < D; ++d) {
        float p = static_cast<float>(phi[phi_off + d]);
        float c = C[C_off + d];
        float s = S[S_off + d * d_v + dv];
        num += p * s;
        den += p * c;
    }
    den = fmaxf(den, EPS);
    out[b * (L * d_v) + l * d_v + dv] = static_cast<scalar_t>(num / den);
}


// ═══════════════════════════════════════════════════════════════════════════════
// Host wrappers
// ═══════════════════════════════════════════════════════════════════════════════

torch::Tensor linear_kernels_attn_causal_cuda(
    const torch::Tensor& phi,
    const torch::Tensor& psi,
    const torch::Tensor& V
) {
    TORCH_CHECK(phi.dim() >= 3, "phi must have >=3 dims (…, L, D)");
    TORCH_CHECK(psi.dim() >= 3, "psi must have >=3 dims (…, L, D)");
    TORCH_CHECK(V.dim()   >= 3, "V must have >=3 dims (…, L, d_v)");
    TORCH_CHECK(phi.size(-1) == psi.size(-1), "phi & psi last dim must match");
    TORCH_CHECK(phi.size(-2) == psi.size(-2) && psi.size(-2) == V.size(-2),
                "sequence lengths must match");

    // Flatten batch dims into one
    auto phi_f = phi.reshape({-1, phi.size(-2), phi.size(-1)}).contiguous();
    auto psi_f = psi.reshape({-1, psi.size(-2), psi.size(-1)}).contiguous();
    auto V_f   = V.reshape({-1, V.size(-2), V.size(-1)}).contiguous();

    int B   = static_cast<int>(phi_f.size(0));
    int L   = static_cast<int>(phi_f.size(1));
    int D   = static_cast<int>(phi_f.size(2));
    int d_v = static_cast<int>(V_f.size(2));

    auto out = torch::empty({B, L, d_v}, phi_f.options());

    int bdim = std::min(D, 256);
    int nw   = (bdim + WARP_SZ - 1) / WARP_SZ;
    size_t shm = 2 * nw * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        phi_f.scalar_type(), "linear_kernels_attn_causal_cuda", ([&] {
            linear_kernels_attn_causal_kernel<scalar_t>
                <<<dim3(B, d_v), dim3(bdim), shm>>>(
                    phi_f.data_ptr<scalar_t>(),
                    psi_f.data_ptr<scalar_t>(),
                    V_f.data_ptr<scalar_t>(),
                    out.data_ptr<scalar_t>(),
                    B, L, D, d_v);
        })
    );
    CUDA_KERNEL_CHECK();

    auto out_shape = phi.sizes().vec();
    out_shape.back() = d_v;
    return out.reshape(out_shape);
}


torch::Tensor linear_kernels_attn_cuda(
    const torch::Tensor& phi,
    const torch::Tensor& psi,
    const torch::Tensor& V
) {
    TORCH_CHECK(phi.dim() >= 3, "phi must have >=3 dims");
    TORCH_CHECK(psi.dim() >= 3, "psi must have >=3 dims");
    TORCH_CHECK(V.dim()   >= 3, "V must have >=3 dims");
    TORCH_CHECK(phi.size(-1) == psi.size(-1), "phi & psi last dim must match");
    TORCH_CHECK(phi.size(-2) == psi.size(-2) && psi.size(-2) == V.size(-2),
                "sequence lengths must match");

    auto phi_f = phi.reshape({-1, phi.size(-2), phi.size(-1)}).contiguous();
    auto psi_f = psi.reshape({-1, psi.size(-2), psi.size(-1)}).contiguous();
    auto V_f   = V.reshape({-1, V.size(-2), V.size(-1)}).contiguous();

    int B   = static_cast<int>(phi_f.size(0));
    int L   = static_cast<int>(phi_f.size(1));
    int D   = static_cast<int>(phi_f.size(2));
    int d_v = static_cast<int>(V_f.size(2));

    auto opts_f = phi_f.options().dtype(torch::kFloat32);

    // Stage 1: C = sum(psi, dim=1)  — use PyTorch (fast & correct)
    auto C = psi_f.sum(/*dim=*/1).to(torch::kFloat32);   // (B, D)

    // Stage 2: S = sum(psi ⊗ V, dim=1)  — custom reduction
    auto S = torch::empty({B, D, d_v}, opts_f);
    int bdim_S = std::min(d_v, 256);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        phi_f.scalar_type(), "compute_S_cuda", ([&] {
            compute_S_kernel<scalar_t>
                <<<dim3(B, D), dim3(bdim_S)>>>(
                    psi_f.data_ptr<scalar_t>(),
                    V_f.data_ptr<scalar_t>(),
                    S.data_ptr<float>(),
                    B, L, D, d_v);
        })
    );
    CUDA_KERNEL_CHECK();

    // Stage 3: output from C and S
    auto out = torch::empty({B, L, d_v}, phi_f.options());

    constexpr int BLK_DV = 64;
    dim3 out_grid(B * L, (d_v + BLK_DV - 1) / BLK_DV);
    dim3 out_block(std::min(d_v, BLK_DV));

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        phi_f.scalar_type(), "linear_kernels_attn_output_cuda", ([&] {
            linear_kernels_attn_output_kernel<scalar_t>
                <<<out_grid, out_block>>>(
                    phi_f.data_ptr<scalar_t>(),
                    C.data_ptr<float>(),
                    S.data_ptr<float>(),
                    out.data_ptr<scalar_t>(),
                    B, L, D, d_v);
        })
    );
    CUDA_KERNEL_CHECK();

    auto out_shape = phi.sizes().vec();
    out_shape.back() = d_v;
    return out.reshape(out_shape);
}


// ═══════════════════════════════════════════════════════════════════════════════
// Module registration
// ═══════════════════════════════════════════════════════════════════════════════

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("linear_kernels_attn_causal", &linear_kernels_attn_causal_cuda,
          "Causal linear-kernel attention (fused CUDA)");
    m.def("linear_kernels_attn", &linear_kernels_attn_cuda,
          "Non-causal linear-kernel attention (CUDA, two-stage)");
}
