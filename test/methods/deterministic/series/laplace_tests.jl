@testitem "Laplace approximation method" begin
    using Test, Distributions
    using ExpectationApproximations
    import SpecialFunctions: beta
    import ExpectationApproximations: Laplace, approximate_expectation

    @testset "Laplace `approximate_expectation` tests: with Gaussian forms" begin
        q        = Normal(3, 1)
        qdensity = x -> -x^2
        brtol    = 0.01
        m        = approximate_expectation(Laplace(1, 1, 3), q, (x) -> x)
        @test isapprox(m, 3.0, rtol = brtol)
        @test isapprox(approximate_expectation(Laplace(1, 1, 3), q, (x) -> 1), 1.0, rtol = brtol)
        @test isapprox(approximate_expectation(Laplace(1, 1, 0), qdensity, (x) -> 1), sqrt(pi), rtol = brtol)
    end

    @testset "Laplace 'approximate_expectation' tests: with Beta forms" begin
        q        = Beta(4, 3)
        qdensity = x -> log(x^(4 - 1) * (1 - x)^(3 - 1))
        brtol    = 0.2
        point    = mode(q)
        m        = approximate_expectation(Laplace(1, 1, point), q, (x) -> x)
        @test isapprox(m, mean(q), atol = brtol)
        @test isapprox(approximate_expectation(Laplace(1, 1, point), q, (x) -> 1), 1.0, atol = brtol)
        @test isapprox(approximate_expectation(Laplace(1, 1, point), qdensity, (x) -> 1), beta(4, 3), atol = brtol)
    end
end
