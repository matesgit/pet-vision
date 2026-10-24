| run | model | pretrained | frozen | augment | epochs | val acc | test acc | macro F1 | params (M) | size (MB) | latency (ms) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| resnet18_linear | resnet18 | True | True | basic | 10 | 0.909 | 0.889 | 0.888 | 11.2 | 42.7 | 4.7 |
| resnet18_ft_strong | resnet18 | True | False | strong | 10 | 0.917 | 0.889 | 0.887 | 11.2 | 42.7 | 4.6 |
| resnet18_ft_noaug | resnet18 | True | False | none | 10 | 0.906 | 0.884 | 0.880 | 11.2 | 42.7 | 3.2 |
| resnet18_ft_basic | resnet18 | True | False | basic | 10 | 0.926 | 0.884 | 0.883 | 11.2 | 42.7 | 3.2 |
| mobilenet_ft_basic | mobilenet_v3_small | True | False | basic | 10 | 0.871 | 0.850 | 0.848 | 1.6 | 6.0 | 4.9 |
| test | resnet18 | True | False | basic | 1 | 0.817 | 0.775 | 0.765 | 11.2 | 42.7 | 2.9 |
| scratch_cnn | simple_cnn | False | False | basic | 30 | 0.447 | 0.353 | 0.342 | 1.2 | 4.5 | 2.8 |