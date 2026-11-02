@testitem "loading registers the backends with the global registry" tags = [:dynamic] setup = [Helpers] begin
    using Distributions, ExponentialFamily

    router = P.DynamicRouter()
    @test P.which_backend(router, log, Gamma(2.0, 1.0)) === B.ClosedForms()
    @test P.which_backend(router, sin, NormalMeanVariance(0.0, 1.0)) isa B.GaussHermite
    @test P.which_backend(router, sin, Gamma(2.0, 1.0)) isa B.QuantileLegendre

    frozen = P.FirstOf(router)
    @test @inferred(P.expectation(frozen, log, Gamma(2.0, 1.0))) == P.expectation(router, log, Gamma(2.0, 1.0))
end
