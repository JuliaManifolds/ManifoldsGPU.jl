# ManifoldsGPU

```@docs
ManifoldsGPU.ManifoldsGPU
```

## Benchmarks

Device: NVIDIA GeForce RTX 5070 Ti, eltype: Float32/ComplexF32

| Manifold | Operation | CPU median [ms] | GPU median [ms] | Speedup CPU/GPU | Relative error |
| --- | --- | ---: | ---: | ---: | ---: |
| Euclidean(32, 16, 2048) | exp | 0.35 | 0.16 | 2.17 | 0.0 |
| Euclidean(32, 16, 2048) | log! | 0.35 | 0.17 | 2.08 | 0.0 |
| Euclidean(32, 16, 2048) | inner | 0.19 | 0.14 | 1.4 | 1.871e-7 |
| Euclidean(32, 16, 2048) | norm | 0.13 | 0.17 | 0.78 | 8.423e-8 |
| Euclidean(32, 16, 2048) | project! | 0.24 | 0.13 | 1.85 | 0.0 |
| PowerManifold(Sphere(31), 2048) | exp | 0.05 | 37.98 | 0.0 | 6.877e-8 |
| PowerManifold(Sphere(31), 2048) | log! | 0.09 | 71.56 | 0.0 | 4.262e-8 |
| PowerManifold(Sphere(31), 2048) | inner | 0.02 | 0.12 | 0.14 | 5.86e-7 |
| PowerManifold(Sphere(31), 2048) | norm | 0.02 | 0.12 | 0.15 | 1.064e-7 |
| PowerManifold(Sphere(31), 2048) | project! | 0.03 | 36.11 | 0.0 | 2.813e-8 |
| PowerManifold(Rotations(32), 2048) | exp | 36.47 | 2.36 | 15.48 | 2.594e-6 |
| PowerManifold(Rotations(32), 2048) | log! | 556.26 | 73.27 | 7.59 | 9.157e-5 |
| PowerManifold(Rotations(32), 2048) | inner | 0.39 | 0.14 | 2.87 | 4.908e-6 |
| PowerManifold(Rotations(32), 2048) | norm | 1.41 | 0.14 | 10.1 | 1.109e-6 |
| PowerManifold(Rotations(32), 2048) | project! | 21.82 | 0.22 | 98.05 | 3.644e-7 |
| PowerManifold(Rotations(32), 2048) | retract_fused!(PolarRetraction) | 116.17 | 4.76 | 24.4 | 2.555e-6 |
| PowerManifold(UnitaryMatrices(32), 2048) | exp | 86.51 | 7.34 | 11.79 | 1.957e-6 |
| PowerManifold(UnitaryMatrices(32), 2048) | log! | 730.58 | 67.51 | 10.82 | 0.0001844 |
| PowerManifold(UnitaryMatrices(32), 2048) | inner | 0.8 | 55.29 | 0.01 | 5.979e-5 |
| PowerManifold(UnitaryMatrices(32), 2048) | norm | 1.79 | 41.11 | 0.04 | 1.516e-6 |
| PowerManifold(UnitaryMatrices(32), 2048) | project! | 31.3 | 0.35 | 89.01 | 5.512e-7 |
| PowerManifold(Grassmann(32, 16), 2048) | exp | 69.74 | 5.36 | 13.01 | 7.023e-5 |
| PowerManifold(Grassmann(32, 16), 2048) | log! | 58.83 | 971.42 | 0.06 | 1.854e-5 |
| PowerManifold(Grassmann(32, 16), 2048) | inner | 0.2 | 0.13 | 1.48 | 1.735e-6 |
| PowerManifold(Grassmann(32, 16), 2048) | norm | 0.8 | 0.14 | 5.78 | 2.772e-7 |
| PowerManifold(Grassmann(32, 16), 2048) | project! | 1.01 | 0.19 | 5.36 | 1.303e-7 |
| PowerManifold(Grassmann(32, 16), 2048) | retract_fused!(PolarRetraction) | 40.94 | 2.91 | 14.07 | 1.338e-6 |
| PowerManifold(Stiefel(32, 16), 2048) | exp(ExponentialRetraction) | 72.86 | 3.39 | 21.47 | 1.164e-6 |
| PowerManifold(Stiefel(32, 16), 2048) | retract_fused!(PolarRetraction) | 43.4 | 2.84 | 15.27 | 1.37e-6 |
