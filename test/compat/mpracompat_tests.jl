@testsnippet CompatSetup begin
    import ExpectationApproximations.MPRACompat as C
    import MessagePassingRulesApproximations as M

    # MessagePassingRulesApproximations 1.0 is the oracle: the same result, in the same shape
    same(a, b; rtol = 1.0e-12) = typeof(a) == typeof(b) && isapprox(a, b; rtol = rtol, atol = rtol)
    same(a::Tuple, b::Tuple; kwargs...) = length(a) == length(b) && all(same(x, y; kwargs...) for (x, y) in zip(a, b))
    same(::Nothing, ::Nothing; kwargs...) = true
    same(a::NamedTuple, b::NamedTuple; kwargs...) = keys(a) == keys(b) && same(Tuple(a), Tuple(b); kwargs...)
end

@testitem "MPRACompat: the unscented transform, as MessagePassingRulesApproximations 1.0" tags = [:compat] setup = [CompatSetup] begin
    m2, V2 = [1.0, -1.0], [2.0 0.5; 0.5 1.0]
    cases = [
        (sin, (0.3,), (0.5,)),
        (x -> [x, x^2], (0.3,), (0.5,)),
        (x -> x[1] * x[2], (m2,), (V2,)),
        (x -> [x[1] * x[2], sin(x[1])], (m2,), (V2,)),
        ((x, y) -> x * y[1] + y[2], (0.3, m2), (0.5, V2)),
        ((x, y) -> [x * y[1], y[2]^2], (0.3, m2), (0.5, V2)),
    ]
    for (f, means, covs) in cases, kwargs in ((;), (alpha = 1.0, beta = 0.0, kappa = 2.0))
        @test same(C.approximate(C.Unscented(; kwargs...), f, means, covs), M.approximate(M.Unscented(; kwargs...), f, means, covs))
        @test same(C.unscented_statistics(C.Unscented(; kwargs...), f, means, covs), M.unscented_statistics(M.Unscented(; kwargs...), f, means, covs))
    end
    @test same(C.sigma_points_weights(C.UT(), 0.3, 0.5), M.sigma_points_weights(M.UT(), 0.3, 0.5))
    @test same(C.sigma_points_weights(C.UT(), m2, V2), M.sigma_points_weights(M.UT(), m2, V2))
    @test same(C.sigma_point_parameters(C.Unscented(), 3), M.sigma_point_parameters(M.Unscented(), 3); rtol = 0)
    @test same(C.sigma_point_parameters(C.Unscented(2; alpha = 1.0), 2), M.sigma_point_parameters(M.Unscented(2; alpha = 1.0), 2); rtol = 0)
    @test C.UnscentedTransform === C.Unscented
    @test C.approximation_name(C.UT()) == M.approximation_name(M.UT())
    @test C.approximation_short_name(C.UT()) == M.approximation_short_name(M.UT())
end

@testitem "MPRACompat: the unscented transform's edge cases" tags = [:compat] setup = [CompatSetup] begin
    # a zero covariance warns and returns a point; an infinite one is a DomainError; a method
    # built for a dimension refuses another
    @test same(@test_logs((:warn,), match_mode = :any, C.unscented_statistics(C.UT(), sin, (0.3,), (0.0,))), (sin(0.3), 0.0, 0.0))
    @test_throws DomainError C.approximate(C.UT(), sin, (0.3,), (Inf,))
    @test_throws DimensionMismatch C.approximate(C.Unscented(2), sin, (0.3,), (0.5,))
end

@testitem "MPRACompat: linearization, as MessagePassingRulesApproximations 1.0" tags = [:compat] setup = [CompatSetup] begin
    m2 = [1.0, -1.0]
    cases = [
        (sin, (0.3,)), (x -> [x, x^2], (0.3,)), (x -> x[1] * x[2], (m2,)), (x -> [x[1] * x[2], sin(x[1])], (m2,)),
        ((x, y) -> x * y[1] + y[2], (0.3, m2)), ((x, y) -> [x * y[1], y[2]^2], (0.3, m2)), ((x, y) -> x .- y, ([1.0, 2.0], [0.5, 0.5])),
    ]
    for (f, x̂) in cases
        @test same(C.approximate(C.Linearization(), f, x̂), M.approximate(M.Linearization(), f, x̂))
        @test same(C.local_linearization(f, x̂), M.local_linearization(f, x̂))
    end
    @test C.approximation_short_name(C.Linearization()) == "LN"
end

@testitem "MPRACompat: Gauss–Hermite cubature, as MessagePassingRulesApproximations 1.0" tags = [:compat] setup = [CompatSetup] begin
    m2, V2 = [1.0, -1.0], [2.0 0.5; 0.5 1.0]
    for p in (3, 11, 31)
        c, o = C.ghcubature(p), M.ghcubature(p)
        @test c isa C.GaussHermiteCubature
        @test C.GaussHermiteCubature(p).piter ≈ o.piter
        @test c.witer ≈ o.witer rtol = 1.0e-8 # this package's weights are the more accurate ones
        @test collect(C.getweights(c, 0.3, 0.5)) ≈ collect(M.getweights(o, 0.3, 0.5)) rtol = 1.0e-8
        @test collect(C.getpoints(c, 0.3, 0.5)) ≈ collect(M.getpoints(o, 0.3, 0.5))
        @test collect(C.getweights(c, m2, V2)) ≈ collect(M.getweights(o, m2, V2)) rtol = 1.0e-8
        @test map(copy, C.getpoints(c, m2, V2)) ≈ map(copy, M.getpoints(o, m2, V2))
        g(x) = exp(-x^2 / 2)
        @test all(C.approximate_meancov(c, g, 0.3, 0.5) .≈ M.approximate_meancov(o, g, 0.3, 0.5))
        h(x) = exp(-sum(abs2, x) / 2)
        cm, cV = C.approximate_meancov(c, h, m2, V2)
        om, oV = M.approximate_meancov(o, h, m2, V2)
        @test cm ≈ om rtol = 1.0e-8
        @test cV ≈ oV rtol = 1.0e-8
    end
    @test sprint(show, C.ghcubature(5)) == sprint(show, M.ghcubature(5))
    @test C.approximation_name(C.ghcubature(5)) == "GaussHermite(5)"
end

@testitem "MPRACompat: a cubature defined elsewhere, through getpoints and getweights" tags = [:compat] setup = [CompatSetup] begin
    # as RxGP does: a method of its own, and approximate_meancov through its points and weights
    struct TwoPoints <: C.AbstractApproximationMethod end
    C.getpoints(::TwoPoints, m::Real, v::Real) = (m - sqrt(v), m + sqrt(v))
    C.getweights(::TwoPoints, m::Real, v::Real) = (0.5, 0.5)
    @test all(C.approximate_meancov(TwoPoints(), x -> 1.0, 0.3, 0.5) .≈ (0.3, 0.5))
end

@testitem "MPRACompat: smoothRTS, as MessagePassingRulesApproximations 1.0" tags = [:compat] setup = [CompatSetup] begin
    @test same(C.smoothRTS(1.0, 2.0, 0.5, 0.3, 1.0, 1.5, 0.7), M.smoothRTS(1.0, 2.0, 0.5, 0.3, 1.0, 1.5, 0.7))
    mt, Vt, Ct = [1.0, -1.0], [2.0 0.5; 0.5 1.0], [0.3 0.1; 0.0 0.2]
    args = (mt, Vt, Ct, [0.1, 0.2], [1.0 0.0; 0.0 1.0], [1.5, -0.5], [0.7 0.0; 0.0 0.4])
    @test same(C.smoothRTS(args...), M.smoothRTS(args...))
    @test same(C.smoothRTS(1.0, Inf, 0.5, 0.3, 1.0, 1.5, 0.7), M.smoothRTS(1.0, Inf, 0.5, 0.3, 1.0, 1.5, 0.7))
end
