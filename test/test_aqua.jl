using Aqua, ManifoldsGPU, Test

@testset "Aqua.jl" begin
    Aqua.test_all(ManifoldsGPU)
end
