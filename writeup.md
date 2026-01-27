### Problem (benchmarking_script): 4 points

(a) Here are my scripts:   
[[python]](cs336_systems/benchmark.py), [[bash]](scripts/run_benchmark.sh)

(b) All experiments were conducted on a single NVIDIA RTX 4090 GPU with 24 GB of memory. The 2.7B model could not be profiled due to GPU out-of-memory (OOM) errors. Results for the remaining model sizes are reported below.
| model_tag   | forward (s)     | forward+backward (s)   |   backward est. (s) |
|:------------|:----------------|:-----------------------|--------------------:|
| small       | 0.0146 ± 0.0013 | 0.0633 ± 0.0047        |              0.0487 |
| medium      | 0.0328 ± 0.0015 | 0.1417 ± 0.0072        |              0.1089 |
| large       | 0.0577 ± 0.0037 | 0.1969 ± 0.0078        |              0.1392 |
| xl          | 0.0971 ± 0.0024 | 0.2889 ± 0.0046        |              0.1919 |
| 2.7B        | 0.1201 ± 0.0016 | nan ± nan              |            nan      |

The standard deviation is relatively small.

(c) If warmup=0:
| model_tag   | forward (s)     | forward+backward (s)   |   backward est. (s) |
|:------------|:----------------|:-----------------------|--------------------:|
| small       | 0.1230 ± 0.3451 | 0.2070 ± 0.3658        |              0.084  |
| medium      | 0.1486 ± 0.3653 | 0.2660 ± 0.3888        |              0.1174 |
| large       | 0.1781 ± 0.3840 | 0.3333 ± 0.4277        |              0.1552 |
| xl          | 0.2159 ± 0.3836 | 0.4245 ± 0.4200        |              0.2085 |
| 2.7B        | 0.2327 ± 0.3555 | nan ± nan              |            nan      |

If warmup=1:
| model_tag   | forward (s)     | forward+backward (s)   |   backward est. (s) |
|:------------|:----------------|:-----------------------|--------------------:|
| small       | 0.0152 ± 0.0020 | 0.0884 ± 0.0072        |              0.0732 |
| medium      | 0.0334 ± 0.0017 | 0.1421 ± 0.0055        |              0.1087 |
| large       | 0.0569 ± 0.0015 | 0.1958 ± 0.0026        |              0.1389 |
| xl          | 0.0989 ± 0.0055 | 0.2898 ± 0.0030        |              0.1908 |
| 2.7B        | 0.1190 ± 0.0016 | nan ± nan              |            nan      |

Without warm-up, the measured mean runtime is higher and shows larger variance, whereas even a single warm-up iteration leads to much more stable and representative measurements.

### Problem (nsys_profile): 5 points

