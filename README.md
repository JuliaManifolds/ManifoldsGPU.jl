# ManifoldsGPU

General GPU/CUDA support for the JuliaManifolds ecosystem.

The package is in early stages of development, and the API is not yet stable.

Notes:

- `exp!` on `PowerManifold(Stiefel(32, 16), 2048)` is about 20x faster on CUDA.
- QR decomposition doesn't seem to be particularly fast on GPU. Q matrix formation can't even be batched as of Feburary 2026.
