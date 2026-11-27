@testitem "GaussJacobi: exact moments of a Beta" tags = [:backends] setup = [EAHelpers] begin
    using Distributions
    @test_throws ArgumentError EA.GaussJacobi(0)
    # E[x^k] = Π (a + i) / (a + b + i)
    for (a, b) in ((0.5, 0.5), (2.0, 5.0), (300.0, 200.0))
        q = Beta(a, b)
        for k in 0:5
            @test P.expectation(EA.GaussJacobi(3), x -> x^k, q) ≈ prod((a + i) / (a + b + i) for i in 0:(k - 1); init = 1.0) rtol = 1.0e-11
        end
    end
    @test P.supports_expectation(EA.GaussJacobi(3), Any, Beta{Float64}) === true
    prep = P.prepare_expectation(EA.GaussJacobi(16), Beta(2.0, 3.0))
    @test allocations(P.expectation, prep, log) == 0
end
