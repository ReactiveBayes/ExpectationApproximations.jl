@testitem "GaussLaguerre: exact moments of a Gamma, any shape" tags = [:backends] setup = [EAHelpers] begin
    using Distributions
    @test_throws ArgumentError EA.GaussLaguerre(0)
    # E[x^k] = θ^k Γ(k + a) / Γ(a), exact up to degree 2n - 1 for every shape, below one too
    for (a, θ) in ((0.3, 2.0), (1.0, 0.5), (2.5, 1.5), (40.0, 0.1))
        q = Gamma(a, θ)
        for k in 0:5
            @test P.expectation(EA.GaussLaguerre(3), x -> x^k, q) ≈ θ^k * prod(a + i for i in 0:(k - 1); init = 1.0) rtol = 1.0e-11
        end
    end
    @test P.expectation(EA.GaussLaguerre(3), x -> x^2, Exponential(2.0)) ≈ 8.0
    # a shape so large that Γ(shape) overflows
    @test P.expectation(EA.GaussLaguerre(8), identity, Gamma(500.0, 0.01)) ≈ 5.0
    @test P.supports(EA.GaussLaguerre(3), Any, Gamma{Float64}) === true
    @test P.supports(EA.GaussLaguerre(3), Any, Normal{Float64}) === false
end

@testitem "GaussLaguerre: duals in the integrand and in the scale" tags = [:backends] setup = [EAHelpers] begin
    using Distributions, ForwardDiff
    # d/dθ E_{Gamma(2, θ)}[x] = 2
    @test ForwardDiff.derivative(θ -> P.expectation(EA.GaussLaguerre(4), identity, Gamma(2.0, θ)), 1.5) ≈ 2.0
    prep = P.prepare_expectation(EA.GaussLaguerre(16), Gamma(2.0, 1.5))
    @test allocations(P.expectation, prep, log) == 0
    # the nodes depend on the shape: its derivative is refused, not silently lost
    @test_throws ArgumentError ForwardDiff.derivative(a -> P.expectation(EA.GaussLaguerre(4), identity, Gamma(a, 1.0)), 2.0)
end
