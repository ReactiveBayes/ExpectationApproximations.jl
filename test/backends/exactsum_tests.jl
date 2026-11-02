@testitem "ExactSum: finite discrete distributions, exactly" tags = [:backends] setup = [EAHelpers] begin
    using Distributions
    @test P.expectation(EA.ExactSum(), x -> x^2, Categorical([0.2, 0.3, 0.5])) ≈ 0.2 + 1.2 + 4.5
    @test P.expectation(EA.ExactSum(), identity, Bernoulli(0.3)) ≈ 0.3
    @test P.expectation(EA.ExactSum(), identity, Binomial(10, 0.3)) ≈ 3.0
    @test P.expectation(EA.ExactSum(), x -> x^2, DiscreteUniform(1, 4)) ≈ (1 + 4 + 9 + 16) / 4
    @test P.expectation(EA.ExactSum(), identity, DiscreteNonParametric([-1.0, 2.0], [0.25, 0.75])) ≈ 1.25
    @test P.expectation(EA.ExactSum(), x -> 2x, Dirac(1.5)) == 3.0
    @test P.expectation(EA.ExactSum(), x -> x .^ 2, P.PointMass([1.0, 2.0])) == [1.0, 4.0]
    for q in (Categorical([0.5, 0.5]), Bernoulli(0.5), Binomial(3, 0.5), Dirac(1.0), P.PointMass(1.0))
        @test P.isexact(EA.ExactSum(), sin, q) === true
    end
    @test P.supports(EA.ExactSum(), Any, Poisson{Float64}) === false
    @test folds_to(P.isexact, (EA.ExactSum, typeof(sin), Categorical{Float64, Vector{Float64}}), true)
    @test allocations(P.expectation, P.prepare_expectation(EA.ExactSum(), Categorical([0.2, 0.8])), sin) == 0
end
