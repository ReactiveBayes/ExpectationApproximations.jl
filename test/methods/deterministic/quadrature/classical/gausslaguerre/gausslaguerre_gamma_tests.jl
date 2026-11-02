@testitem "GaussLaguerre: Basic integrations/approximations when q is Gamma" tags = [:quadrature] begin
    using ExpectationApproximations, Distributions, StableRNGs
    import ExpectationApproximations: ClassicalQuadratureStrategyAutomatic

    irng = StableRNG(123)

    btol(::UseLogWeights, n) = 1.0e-2 / log(n) # LogWeights should have better accuracy overall
    btol(::UseWeights, n) = 1.0e-2
    btol(::ClassicalQuadratureStrategyAutomatic, n) = 1.0e-2

    strategies = (UseLogWeights(), UseWeights(), ClassicalQuadratureStrategyAutomatic())
    npoints = (10, 20, 100, 200)
    qs = (Gamma(20rand(irng) + 1, 20rand(irng) + 1) for _ in 1:50)

    for strategy in strategies, n in npoints, q in qs
        method = ClassicalQuadrature(GaussLaguerre(n), strategy)

        # base rel tolerance
        brtol = btol(strategy, n)

        @test isapprox(approximate(method, q, (x) -> 1), 1.0, rtol = brtol)
        @test isapprox(approximate(method, q, (x) -> -1), -1.0, rtol = brtol)
        @test isapprox(approximate(method, q, LogIntegrand((x) -> 0.0)), 1.0, rtol = brtol)

        @test isapprox(approximate(method, q, (x) -> x), mean(q), rtol = brtol)
        @test isapprox(approximate(method, q, (x) -> -x), -mean(q), rtol = brtol)
        @test isapprox(approximate(method, q, LogIntegrand((x) -> log(x))), mean(q), rtol = brtol)

        @test isapprox(approximate(method, q, (x) -> (x - mean(q))^2), var(q), rtol = brtol)
        @test isapprox(approximate(method, q, (x) -> -(x - mean(q))^2), -var(q), rtol = brtol)
        @test isapprox(approximate(method, q, LogIntegrand((x) -> 2log(abs(x - mean(q))))), var(q), rtol = brtol)
    end
end

@testitem "GaussLaguerre moments calculation when q is Gamma" tags = [:quadrature] begin
    using ExpectationApproximations, Distributions, StableRNGs
    import ExpectationApproximations: ClassicalQuadratureStrategyAutomatic

    btol(::UseLogWeights, n) = 1.0e-1 / log(n) # LogWeights should have better accuracy overall
    btol(::UseWeights, n) = 1.0e-1
    btol(::ClassicalQuadratureStrategyAutomatic, n) = 1.0e-1

    vals = (1.3, 2.1, 10.1)
    npoints = (61, 121, 1121, 2121)
    strategies = (UseLogWeights(), UseWeights(), ClassicalQuadratureStrategyAutomatic())

    for α1 in vals, α2 in vals, β1 in vals, β2 in vals, n in npoints, strategy in strategies
        d1 = Gamma(α1, 1 / β1)
        d2 = Gamma(α2, 1 / β2)

        pα = α1 + α2 - 1
        pβ = β1 + β2
        p = Gamma(pα, 1 / pβ)

        for f in ((x) -> pdf(d2, x), LogIntegrand((x) -> logpdf(d2, x)), d2)
            method = ClassicalQuadrature(GaussLaguerre(n), strategy)
            C = approximate(Expectation(), method, d1, f)
            m = approximate(MeasureAdjustedRawMomentGenerator(Val(1), C), method, d1, f)
            v = approximate(MeasureAdjustedMomentGenerator(Val(2), C, m), method, d1, f)

            # base rel tolerance
            brtol = btol(strategy, n)

            @test isapprox(m, mean(p), rtol = brtol)
            @test isapprox(v, var(p), rtol = brtol)
        end
    end
end
