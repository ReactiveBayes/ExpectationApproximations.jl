@testitem "GaussHermite: Basic integrations/approximations when q is Normal" begin
    using ExpectationApproximations, Distributions, StableRNGs
    import ExpectationApproximations: ClassicalQuadratureStrategyAutomatic

    irng = StableRNG(123)

    btol(::UseLogWeights, n) = 1e-2 / log(n) # LogWeights should have better accuracy overall
    btol(::UseWeights, n) = 1e-2
    btol(::ClassicalQuadratureStrategyAutomatic, n) = 1e-2

    strategies = (UseLogWeights(), UseWeights(), ClassicalQuadratureStrategyAutomatic())
    npoints = (10, 20, 100, 200)
    qs = (Normal(20randn(irng) + 1, 20rand(irng) + 1) for _ in 1:50)

    for strategy in strategies, n in npoints, q in qs
        method = ClassicalQuadrature(GaussHermite(n), strategy)

        # base rel tolerance 
        brtol = btol(strategy, n)

        @test isapprox(approximate(method, q, (x) -> 1), 1.0, rtol = brtol)
        @test isapprox(approximate(method, q, (x) -> -1), -1.0, rtol = brtol)
        @test isapprox(approximate(method, q, LogIntegrand((x) -> 0.0)), 1.0, rtol = brtol)

        @test isapprox(approximate(method, q, (x) -> x), mean(q), rtol = brtol)
        @test isapprox(approximate(method, q, (x) -> -x), -mean(q), rtol = brtol)
        @test isapprox(approximate(method, q, LogIntegrand((x) -> log(Complex(x)))), mean(q), rtol = brtol)

        @test isapprox(approximate(method, q, (x) -> (x - mean(q))^2), var(q), rtol = brtol)
        @test isapprox(approximate(method, q, (x) -> -(x - mean(q))^2), -var(q), rtol = brtol)
        @test isapprox(approximate(method, q, LogIntegrand((x) -> 2log(abs(x - mean(q))))), var(q), rtol = brtol)
    end
end

@testitem "GaussHermite moments calculation when q is Normal" begin
    using ExpectationApproximations, Distributions, StableRNGs
    import ExpectationApproximations: ClassicalQuadratureStrategyAutomatic

    btol(::UseLogWeights, n) = 5e-1 / log(n) # LogWeights should have better accuracy overall
    btol(::UseWeights, n) = 5e-1
    btol(::ClassicalQuadratureStrategyAutomatic, n) = 5e-1

    m1s = (1.3, -2.1, 10.1)
    m2s = (1.3, -2.1, 10.1)
    v1s = (0.1, 1.0, 4.0)
    v2s = (0.1, 1.0, 4.0)
    npoints = (121, 1121, 2121)
    strategies = (UseLogWeights(), UseWeights(), ClassicalQuadratureStrategyAutomatic())

    for m1 in m1s, m2 in m2s, v1 in v1s, v2 in v2s, n in npoints, strategy in strategies
        d1 = Normal(m1, sqrt(v1))
        d2 = Normal(m2, sqrt(v2))

        m = (m1 * v2 + m2 * v1) / (v1 + v2)
        v = (v1 * v2) / (v1 + v2)
        p = Normal(m, sqrt(v))

        for f in ((x) -> pdf(d2, x), LogIntegrand((x) -> logpdf(d2, x)), d2)
            method = ClassicalQuadrature(GaussHermite(n), strategy)
            C      = approximate(Expectation(), method, d1, f)
            m      = approximate(RawMoment(Val(1), C), method, d1, f)
            v      = approximate(CentralMoment(Val(2), C, m), method, d1, f)

            # base rel tolerance 
            brtol = btol(strategy, n)

            @test isapprox(m, mean(p), rtol = brtol)
            @test isapprox(v, var(p), rtol = brtol)
        end
    end
end
