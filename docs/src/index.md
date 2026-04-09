# ManifoldsGPU

```@docs
ManifoldsGPU.ManifoldsGPU
```

## Benchmarks

Device: NVIDIA GeForce RTX 5070 Ti, eltype: Float32/ComplexF32

| Manifold | Operation | CPU median [ms] | GPU median [ms] | Speedup CPU/GPU | Relative error |
| --- | --- | ---: | ---: | ---: | ---: |
| Euclidean(32, 16, 2048) | exp | 0.36 | 0.17 | 2.17 | 0.0 |
| Euclidean(32, 16, 2048) | log! | 0.36 | 0.16 | 2.19 | 0.0 |
| Euclidean(32, 16, 2048) | inner | 0.19 | 0.14 | 1.36 | 9.357e-8 |
| Euclidean(32, 16, 2048) | norm | 0.12 | 0.16 | 0.76 | 8.423e-8 |
| Euclidean(32, 16, 2048) | project! | 0.25 | 0.13 | 1.93 | 0.0 |
| PowerManifold(Sphere(31), 2048) | exp | 0.05 | 46.57 | 0.0 | 6.877e-8 |
| PowerManifold(Sphere(31), 2048) | log! | 0.16 | 82.22 | 0.0 | 4.262e-8 |
| PowerManifold(Sphere(31), 2048) | inner | 0.02 | 0.12 | 0.13 | 5.86e-7 |
| PowerManifold(Sphere(31), 2048) | norm | 0.02 | 0.12 | 0.15 | 1.064e-7 |
| PowerManifold(Sphere(31), 2048) | project! | 0.03 | 41.71 | 0.0 | 2.813e-8 |
| PowerManifold(Rotations(32), 2048) | exp | 36.47 | 2.36 | 15.47 | 2.594e-6 |
| PowerManifold(Rotations(32), 2048) | log! | 570.16 | 78.98 | 7.22 | 9.157e-5 |
| PowerManifold(Rotations(32), 2048) | inner | 0.38 | 0.15 | 2.58 | 4.708e-6 |
| PowerManifold(Rotations(32), 2048) | norm | 1.2 | 0.15 | 7.82 | 9.132e-7 |
| PowerManifold(Rotations(32), 2048) | project! | 21.69 | 0.23 | 94.53 | 3.644e-7 |
| PowerManifold(Rotations(32), 2048) | retract_fused!(PolarRetraction) | 116.78 | 5.21 | 22.41 | 2.555e-6 |
| PowerManifold(UnitaryMatrices(32), 2048) | exp | 85.95 | 7.66 | 11.21 | 1.957e-6 |
| PowerManifold(UnitaryMatrices(32), 2048) | log! | 739.96 | 69.31 | 10.68 | 0.0001844 |
| PowerManifold(UnitaryMatrices(32), 2048) | inner | 0.76 | 58.76 | 0.01 | 5.979e-5 |
| PowerManifold(UnitaryMatrices(32), 2048) | norm | 1.73 | 45.7 | 0.04 | 1.516e-6 |
| PowerManifold(UnitaryMatrices(32), 2048) | project! | 31.55 | 0.36 | 88.48 | 5.512e-7 |
| PowerManifold(Grassmann(32, 16), 2048) | exp | 69.77 | 5.36 | 13.01 | 7.023e-5 |
| PowerManifold(Grassmann(32, 16), 2048) | log! | 57.62 | 3.3 | 17.48 | 2.332e-5 |
| PowerManifold(Grassmann(32, 16), 2048) | inner | 0.19 | 0.14 | 1.41 | 8.675e-7 |
| PowerManifold(Grassmann(32, 16), 2048) | norm | 0.86 | 0.13 | 6.58 | 2.772e-7 |
| PowerManifold(Grassmann(32, 16), 2048) | project! | 0.96 | 0.19 | 4.97 | 1.303e-7 |
| PowerManifold(Grassmann(32, 16), 2048) | retract_fused!(PolarRetraction) | 40.89 | 2.69 | 15.18 | 1.338e-6 |
| PowerManifold(Stiefel(32, 16), 2048) | exp(ExponentialRetraction) | 70.67 | 3.52 | 20.07 | 1.164e-6 |
| PowerManifold(Stiefel(32, 16), 2048) | retract_fused!(PolarRetraction) | 43.1 | 2.85 | 15.13 | 1.37e-6 |
