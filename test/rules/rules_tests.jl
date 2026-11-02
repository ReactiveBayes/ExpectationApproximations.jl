@testsnippet RulesSetup begin
    using FastGaussQuadrature: FastGaussQuadrature
    using SpecialFunctions: gamma, beta
    const EA = ExpectationApproximations

    # Nodes against a reference, relative to the size of the node; weights relative to their own
    # size, wherever they are above underflow
    function agrees(rule, reference; nodes = 1.0e-13, weights = 1.0e-11)
        x, w = rule
        xr, wr = reference
        length(x) == length(xr) || return false
        nodes_ok = all(abs.(x .- xr) .<= nodes .* max.(1, abs.(xr)))
        visible = wr .> 1.0e-280
        weights_ok = all(abs.(w[visible] .- wr[visible]) .<= weights .* wr[visible])
        return nodes_ok && weights_ok
    end

    # Σ wᵢ f(xᵢ)
    integrate(f, (x, w)) = sum(w .* f.(x))

    # The same recurrence in 256-bit arithmetic, Newton-polished from the Float64 nodes: checks
    # the precision of a rule, while FastGaussQuadrature and exactness check its correctness
    function bigweights((x, w), a, b, μ0)
        setprecision(256) do
            xs = BigFloat.(x)
            for i in eachindex(xs), _ in 1:6
                value, derivative = EA.recurrence(xs[i], a, b, μ0)
                xs[i] -= value / derivative
            end
            return xs, [EA.christoffel(xi, a, b, μ0) for xi in xs]
        end
    end
    function precise((x, w), (xb, wb); tolerance = 1.0e-13)
        return all(abs.(x .- xb) .<= tolerance .* max.(1, abs.(xb))) &&
            all(abs.(w .- wb) .<= tolerance .* wb)
    end
end

@testitem "gausshermite: agrees with FastGaussQuadrature" tags = [:rules] setup = [RulesSetup] begin
    # FastGaussQuadrature's own weights are accurate to about 1e-9 for the smallest of them at
    # n = 50 (see the precision test below), so the comparison allows that much
    for n in (1, 2, 3, 5, 10, 20, 50, 100)
        @test agrees(EA.gausshermite(n), FastGaussQuadrature.gausshermite(n); weights = 1.0e-9)
    end
end

@testitem "rules: precise to 1e-13 against 256-bit arithmetic" tags = [:rules] setup = [RulesSetup] begin
    setprecision(256) do
        for n in (5, 20, 50, 100)
            a, b = zeros(BigFloat, n), [sqrt(BigFloat(k) / 2) for k in 1:(n - 1)]
            rule = EA.gausshermite(n)
            @test precise(rule, bigweights(rule, a, b, sqrt(BigFloat(π))))

            α = BigFloat(1) / 2
            a, b = [2k + α + 1 for k in 0:(n - 1)], [sqrt(k * (k + α)) for k in 1:(n - 1)]
            rule = EA.gausslaguerre(n, 0.5)
            @test precise(rule, bigweights(rule, a, b, gamma(α + 1)))
        end
    end
end

@testitem "gausshermite: exact up to degree 2n - 1" tags = [:rules] setup = [RulesSetup] begin
    # ∫ x^k exp(-x²) dx = Γ((k + 1) / 2) for even k, zero for odd k
    for n in (1, 3, 8), k in 0:(2n - 1)
        exact = iseven(k) ? gamma((k + 1) / 2) : 0.0
        @test integrate(x -> x^k, EA.gausshermite(n)) ≈ exact atol = 1.0e-12 * max(1, exact)
    end
end

@testitem "gausslaguerre: agrees with FastGaussQuadrature, generalised" tags = [:rules] setup = [RulesSetup] begin
    for n in (1, 2, 3, 5, 10, 20, 50, 100), α in (0.0, -0.5, 0.7, 3.0)
        @test agrees(EA.gausslaguerre(n, α), FastGaussQuadrature.gausslaguerre(n, α); weights = 1.0e-10)
    end
    @test agrees(EA.gausslaguerre(10), FastGaussQuadrature.gausslaguerre(10))
end

@testitem "gausslaguerre: exact up to degree 2n - 1" tags = [:rules] setup = [RulesSetup] begin
    # ∫₀^∞ x^k x^α exp(-x) dx = Γ(k + α + 1)
    for n in (1, 3, 8), α in (-0.5, 0.0, 2.5), k in 0:(2n - 1)
        @test integrate(x -> x^k, EA.gausslaguerre(n, α)) ≈ gamma(k + α + 1) rtol = 1.0e-11
    end
end

@testitem "gaussjacobi and gausslegendre: agree with FastGaussQuadrature" tags = [:rules] setup = [RulesSetup] begin
    for n in (1, 2, 3, 5, 10, 20, 50, 100)
        @test agrees(EA.gausslegendre(n), FastGaussQuadrature.gausslegendre(n))
        for (α, β) in ((0.0, 0.0), (-0.5, -0.5), (1.0, 2.0), (2.5, -0.3))
            @test agrees(EA.gaussjacobi(n, α, β), FastGaussQuadrature.gaussjacobi(n, α, β); weights = 1.0e-10)
        end
    end
end

@testitem "gaussjacobi: exact up to degree 2n - 1" tags = [:rules] setup = [RulesSetup] begin
    # with x = 2u - 1: ∫₋₁¹ x^k (1 - x)^α (1 + x)^β dx = 2^(α+β+1) E[(2u - 1)^k] B(β+1, α+1),
    # u ~ Beta(β + 1, α + 1); the expectation of the polynomial follows from Beta moments
    # the alternating sum cancels, so it is evaluated in 256-bit arithmetic
    betamoment(j, a, b) = prod((a + i) / (a + b + i) for i in 0:(j - 1); init = one(a))
    setprecision(256) do
        for n in (1, 3, 8), (α, β) in ((0.0, 0.0), (1.5, 0.5)), k in 0:(2n - 1)
            A, B = BigFloat(α), BigFloat(β)
            expectation = sum(binomial(k, j) * BigFloat(2)^j * (-1)^(k - j) * betamoment(j, B + 1, A + 1) for j in 0:k)
            exact = Float64(2^(A + B + 1) * beta(B + 1, A + 1) * expectation)
            @test integrate(x -> x^k, EA.gaussjacobi(n, α, β)) ≈ exact atol = 1.0e-13
        end
    end
end

@testitem "rules: sorted nodes inside the domain, positive weights, Float64" tags = [:rules] setup = [RulesSetup] begin
    for rule in (EA.gausshermite(9), EA.gausslaguerre(9, 0.5), EA.gaussjacobi(9, 0.5, 1.5), EA.gausslegendre(9))
        x, w = rule
        @test issorted(x)
        @test all(>(0), w)
        @test x isa Vector{Float64}
        @test w isa Vector{Float64}
    end
    @test all(>(0), first(EA.gausslaguerre(9, 0.5)))
    @test all(x -> -1 < x < 1, first(EA.gaussjacobi(9, 0.5, 1.5)))
    @test @inferred(EA.gausshermite(4)) isa Tuple{Vector{Float64}, Vector{Float64}}
    @test @inferred(EA.gausslaguerre(4, 1.0)) isa Tuple{Vector{Float64}, Vector{Float64}}
    @test @inferred(EA.gaussjacobi(4, 1.0, 1.0)) isa Tuple{Vector{Float64}, Vector{Float64}}
end

@testitem "rules: invalid arguments say what is wrong" tags = [:rules] setup = [RulesSetup] begin
    @test_throws ArgumentError EA.gausshermite(0)
    @test_throws ArgumentError EA.gausslaguerre(0)
    @test_throws ArgumentError EA.gausslaguerre(3, -1.0)
    @test_throws ArgumentError EA.gaussjacobi(3, -1.0, 0.0)
    @test_throws ArgumentError EA.gaussjacobi(3, 0.0, -1.5)
    @test_throws ArgumentError EA.gausslegendre(-2)
end
