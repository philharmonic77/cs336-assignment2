### Problem (benchmarking_script): 4 points

(a) Here are my scripts:   
[[python]](cs336_systems/benchmark.py), [[bash]](cs336_systems/scripts/run_benchmark.sh)

(b) All experiments were conducted on a single NVIDIA RTX 4090 GPU with 24 GB of memory. The 2.7B model could not be profiled due to GPU out-of-memory (OOM) errors. Results for the remaining model sizes are reported below.
| model_tag   | forward (s)     | forward+backward (s)   |   backward est. (s) |
|:------------|:----------------|:-----------------------|--------------------:|
| small       | 0.0268 ± 0.0015 | 0.0987 ± 0.0039        |              0.0719 |
| medium      | 0.0527 ± 0.0017 | 0.1441 ± 0.0128        |              0.0914 |
| large       | 0.0880 ± 0.0050 | 0.2515 ± 0.0188        |              0.1635 |
| xl          | 0.1344 ± 0.0031 | 0.3287 ± 0.0296        |              0.1943 |

The standard deviation is relatively small.