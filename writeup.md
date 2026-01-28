### Problem (benchmarking_script): 4 points

(a) Here are my scripts used for both benchmark calculating and nsys profiling:   
[[python]](cs336_systems/nsys_profile.py), [[bash]](scripts/run_nsys_profile.sh)

(b) All experiments were conducted on a single NVIDIA RTX 4090 GPU with 24 GB of memory, using a context length of 128. A complete training step uses AdamW as the optimizer. Results for the remaining model sizes are reported below.

| model_tag   | forward(infer) /s   | train_step /s   |
|:------------|:--------------------|:----------------|
| small       | 0.0297 ± 0.0014     | 0.1579 ± 0.0056 |
| medium      | 0.0564 ± 0.0002     | 0.2633 ± 0.0070 |
| large       | 0.0839 ± 0.0012     | 0.3663 ± 0.0155 |
| xl          | 0.1341 ± 0.0031     | OOM             |
| 2.7B        | 0.1462 ± 0.0023     | OOM             |

The standard deviation is relatively small.

(c) If warmup=0:
| model_tag   | forward(infer) /s   | train_step /s   |
|:------------|:--------------------|:----------------|
| small       | 0.0637 ± 0.1131     | 0.2052 ± 0.1647 |
| medium      | 0.0957 ± 0.1176     | 0.3197 ± 0.1738 |
| large       | 0.1242 ± 0.1174     | 0.4391 ± 0.1941 |
| xl          | 0.1738 ± 0.1237     | OOM             |
| 2.7B        | 0.1800 ± 0.1180     | OOM             |

If warmup=1:
| model_tag   | forward(infer) /s   | train_step /s   |
|:------------|:--------------------|:----------------|
| small       | 0.0302 ± 0.0026     | 0.1541 ± 0.0033 |
| medium      | 0.0593 ± 0.0007     | 0.2713 ± 0.0037 |
| large       | 0.0893 ± 0.0023     | 0.4032 ± 0.0137 |
| xl          | 0.1341 ± 0.0029     | OOM             |
| 2.7B        | 0.1417 ± 0.0061     | OOM             |

Without warm-up, the measured mean runtime is higher and shows larger variance, whereas even a single warm-up iteration leads to much more stable and representative measurements.

### Problem (nsys_profile): 5 points
Also, all experiments were conducted on a single NVIDIA RTX 4090 GPU with 24 GB of memory, using AdamW as optimizer. Below is the result.

| model_tag | context_len | forward(inference)/s | forward(train)/s | backward/s | optimizer_step/s | train_step/s |
|---|---:|---:|---:|---:|---:|---:|
| small | 128 | 0.0296 | 0.0364 | 0.0874 | 0.0324 | 0.1581 |
| medium | 128 | 0.0562 | 0.0661 | 0.1465 | 0.0468 | 0.2634 |
| large | 128 | 0.0834 | 0.0987 | 0.2013 | 0.0576 | 0.3664 |
| xl | 128 | 0.1332 |  |  |  |  |
| 2.7B | 128 | 0.1438 |  |  |  |  |
| small | 256 | 0.0297 | 0.0389 | 0.0873 | 0.0297 | 0.1576 |
| medium | 256 | 0.0619 | 0.0666 | 0.1442 | 0.0481 | 0.2626 |
| large | 256 | 0.1097 | 0.1267 | 0.2152 | 0.0576 | 0.4081 |
| xl | 256 | 0.1858 |  |  |  |  |
| 2.7B | 256 | 0.2262 |  |  |  |  |
| small | 512 | 0.0331 | 0.0391 | 0.0878 | 0.0325 | 0.1613 |
| medium | 512 | 0.0959 | 0.0989 | 0.1463 | 0.0479 | 0.3008 |
| large | 512 | 0.2015 |  |  |  |  |
| xl | 512 | 0.3764 |  |  |  |  |
| 2.7B | 512 | 0.4476 |  |  |  |  |
| small | 1024 | 0.0819 | 0.0843 | 0.0851 | 0.0696 | 0.2562 |
| medium | 1024 | 0.2344 |  |  |  |  |
| large | 1024 | 0.4554 |  |  |  |  |
| xl | 1024 | 0.8444 |  |  |  |  |
| 2.7B | 1024 | 1.0402 |  |  |  |  |

(a) The forward-pass runtime measured by Nsight Systems closely matches the wall-clock measurements obtained using Python’s timeit, indicating that the benchmark accurately captures steady-state GPU execution time.

(b) Using large model with 256 context length as emample:
- forward(inference)

![](assets/nsys_profile_large_256_forward.png)

- backward

![](assets/nsys_profile_large_256_backward.png)

- adamW

![](assets/nsys_profile_large_256_adamw.png)

During the forward pass, the CUDA kernel that dominates cumulative GPU time is the GEMM kernel (e.g., ampere_sgemm_128x64_tn), corresponding to the linear layers in attention and MLPs, and it is invoked on the order of thousands of times per forward pass . When running both forward and backward, GEMM kernels still dominate overall runtime, but backward-specific GEMMs (gradient w.r.t. weights/activations) and additional elementwise kernels take a larger fraction, so the single most expensive kernel may differ from the forward-only case.

(c) Besides matrix multiplications, a non-trivial portion of the forward-pass CUDA runtime is spent on elementwise kernels from SwiGLU activations (SiLU and gating), attention softmax and other reduction-based operations, as well as LayerNorm kernels that compute mean/variance and normalization.

(d) While GEMM kernels dominate inference, their relative contribution decreases in a full training step, as backward propagation adds multiple new GEMM variants and the AdamW optimizer is dominated by vectorized elementwise kernels such as addcmul, addcdiv, and sqrt.
- 10 forward(inference) pass

![](assets/nsys_profile_large_256_forward.png)

- 10 compelete training steps with adamW

![](assets/nsys_profile_large_256_train.png)

(e)	For the large model with context length 256, the FLOPs are:
-  2BS^2d_h for computing attention scores
- 4BS^2 for softmax
- 2BS^2d_h for the final matmul

giving an approximate FLOP ratio of **64 : 1 : 64**.


In contrast, Nsight Systems reports runtimes of 6.27 ms (scores), 4.09 ms (softmax), and 3.79 ms (final matmul), because:
- softmax is implemented as multiple unfused element-wise kernels with high memory traffic
- computing attention scores additionally includes reshapes and scaling operations beyond pure matrix multiplication.

![](assets/nsys_profile_large_256_forward_annotated.png)

Here is the script used: [[bash]](scripts/run_nsys_profile_annotated_attention.sh)


### Problem (mixed_precision_accumulation): 1 point

This experiment shows that numerical error is dominated by the precision used for accumulation rather than the precision of individual operands. Accumulating FP16 values in FP16 leads to large systematic error due to repeated rounding, while accumulating the same FP16 values in FP32 significantly improves numerical accuracy. This motivates mixed-precision training, where compute-heavy operations use low precision but reductions and accumulations are kept in FP32.

![](assets/mixed_precision_accumulation.png)


### Problem (benchmarking_mixed_precision): 2 points
(a) The data types for each of the components:  
- parameters: FP32
- fc1 output: FP16
- layernorm output: FP16 (with FP32 internal accumulation/reduction)
- logits (fc2 output): FP16
- loss: FP32
- gradients: FP32

(b) Below is the formula：

1. Calculate RMS:
$\mathrm{RMS}(\mathbf{x}) = \sqrt{\frac{1}{d} \sum_{i=1}^{d} x_i^2 + \epsilon}$

2. Normalize: 
$\hat{x}_i = \frac{x_i}{\mathrm{RMS}(\mathbf{x})}$

3. Scaling:
$y_i = \gamma_i \cdot \hat{x}_i$

Squaring and accumulation operations in layer normalization are sensitive to numerical range, since intermediate values can become large and overflow when using FP16. FP16 has a limited exponent range, making such reductions unstable. In contrast, BF16 has the same exponent range as FP32, so it is much less prone to overflow, and layer normalization does not require special handling when using BF16.