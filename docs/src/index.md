# ManifoldsGPU

```@docs
ManifoldsGPU.ManifoldsGPU
```

## Benchmarks

=== Markdown summary table ===
Device: NVIDIA GeForce RTX 5070 Ti, eltype: Float32/ComplexF32
| Manifold | Operation | CPU median [ms] | GPU median [ms] | Speedup CPU/GPU | Error |
| --- | --- | ---: | ---: | ---: | ---: |
| Euclidean(32, 16, 2048) | exp | 0.34 | 0.17 | 2.07 | 0.0 |
| Euclidean(32, 16, 2048) | log! | 0.34 | 0.17 | 2.03 | 0.0 |
| Euclidean(32, 16, 2048) | inner | 0.19 | 0.13 | 1.42 | 0.0 |
| Euclidean(32, 16, 2048) | norm | 0.12 | 0.16 | 0.74 | 8.423e-8 |
| Euclidean(32, 16, 2048) | project! | 0.23 | 0.13 | 1.76 | 0.0 |
| PowerManifold(Sphere(31), 2048) | exp | 0.05 | 36.49 | 0.0 | 6.877e-8 |
| PowerManifold(Sphere(31), 2048) | log! | 0.09 | 68.95 | 0.0 | 4.262e-8 |
| PowerManifold(Sphere(31), 2048) | inner | 0.02 | 0.12 | 0.12 | 6.837e-7 |
| PowerManifold(Sphere(31), 2048) | norm | 0.02 | 0.13 | 0.15 | 0.0 |
| PowerManifold(Sphere(31), 2048) | project! | 0.03 | 34.91 | 0.0 | 2.813e-8 |
| PowerManifold(Rotations(32), 2048) | exp | 36.83 | 2.35 | 15.68 | 2.594e-6 |
| PowerManifold(Rotations(32), 2048) | log! | 561.36 | 67.61 | 8.3 | 9.157e-5 |
| PowerManifold(Rotations(32), 2048) | inner | 0.41 | 0.14 | 2.85 | 4.508e-6 |
| PowerManifold(Rotations(32), 2048) | norm | 1.32 | 0.14 | 9.64 | 1.109e-6 |
| PowerManifold(Rotations(32), 2048) | project! | 20.33 | 0.22 | 93.02 | 3.644e-7 |
| PowerManifold(Rotations(32), 2048) | retract_fused!(PolarRetraction) | 116.77 | 4.82 | 24.24 | 2.555e-6 |
| PowerManifold(Rotations(32), 2048) | retract_fused!(QRRetraction) | 90.92 | 0.85 | 106.99 | 3.204e-7 |
| PowerManifold(UnitaryMatrices(32), 2048) | exp | 86.44 | 7.65 | 11.3 | 1.957e-6 |
| PowerManifold(UnitaryMatrices(32), 2048) | log! | 728.19 | 70.03 | 10.4 | 0.0001844 |
| PowerManifold(UnitaryMatrices(32), 2048) | inner | 0.78 | 52.19 | 0.02 | 5.979e-5 |
| PowerManifold(UnitaryMatrices(32), 2048) | norm | 1.76 | 41.09 | 0.04 | 1.516e-6 |
| PowerManifold(UnitaryMatrices(32), 2048) | project! | 31.94 | 0.36 | 89.84 | 5.512e-7 |
| PowerManifold(Grassmann(32, 16), 2048) | exp | 69.47 | 5.23 | 13.27 | 7.023e-5 |
| PowerManifold(Grassmann(32, 16), 2048) | log! | 57.7 | 3.42 | 16.85 | 2.332e-5 |
| PowerManifold(Grassmann(32, 16), 2048) | inner | 0.2 | 0.12 | 1.58 | 4.957e-7 |
| PowerManifold(Grassmann(32, 16), 2048) | norm | 0.79 | 0.13 | 6.18 | 2.772e-7 |
| PowerManifold(Grassmann(32, 16), 2048) | project! | 1.03 | 0.19 | 5.31 | 1.303e-7 |
| PowerManifold(Grassmann(32, 16), 2048) | retract_fused!(PolarRetraction) | 41.38 | 2.75 | 15.06 | 0.0001873 |
| PowerManifold(Grassmann(32, 16), 2048) | retract_fused!(QRRetraction) | 17.85 | 0.7 | 25.33 | 4.623e-5 |
| PowerManifold(Stiefel(32, 16), 2048) | exp(ExponentialRetraction) | 71.95 | 3.65 | 19.71 | 1.164e-6 |
| PowerManifold(Stiefel(32, 16), 2048) | retract_fused!(PolarRetraction) | 43.03 | 3.06 | 14.07 | 1.37e-6 |
| PowerManifold(Stiefel(32, 16), 2048) | retract_fused!(QRRetraction) | 18.38 | 0.71 | 25.88 | 1.885e-7 |