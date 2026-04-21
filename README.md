# ManifoldsGPU

[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://juliamanifolds.github.io/ManifoldsGPU.jl/dev/)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
[![code style: runic](https://img.shields.io/badge/code_style-%E1%9A%B1%E1%9A%A2%E1%9A%BE%E1%9B%81%E1%9A%B2-black)](https://github.com/fredrikekre/Runic.jl)

General GPU/CUDA support for the JuliaManifolds ecosystem.

The package is in early stages of development, and the API is not yet stable.

Notes:

- `exp!` on `PowerManifold(Stiefel(32, 16), 2048)` is about 20x faster on CUDA.
- `PolarRetraction` is about 15x faster on CUDA. Batched SVD seems to work well.
- Detailed benchmarking scripts are in `benchmarks/`.
- QR decomposition doesn't seem to be particularly fast on GPU. Q matrix formation can't even be batched as of February 2026.
