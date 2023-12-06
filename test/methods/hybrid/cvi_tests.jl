@testitem "CVI" begin
    include("./hybrid_setuptests.jl")
    @testset "Constructor CVI method" begin
        cvi_method = CVI(StableRNG(42), 1000, Optimisers.Adam(), ForwardDiffGrad(), 900, Val(true), true)
        @test cvi_method.n_iterations == 1000
        @test cvi_method.n_gradpoints == 900
        @test cvi_method.enforce_proper_messages == Val(true)
        @test cvi_method.warn == true
    end

    @testset "Constructor CVI Factor" begin
        cvi_factor = CVIFactor(ExponentialFamily.ExponentialFamilyDistribution(Bernoulli, [0.7]), ExponentialFamily.ExponentialFamilyDistribution(Bernoulli, [0.9]))

        @test ExpectationApproximations.get_exponential_family(cvi_factor) == ExponentialFamily.ExponentialFamilyDistribution(Bernoulli, [0.7])
        @test ExpectationApproximations.get_positive_measure(cvi_factor) == ExponentialFamily.ExponentialFamilyDistribution(Bernoulli, [0.9])
    end
end

@testitem "CVI objective" begin
    include("./hybrid_setuptests.jl")

    function generate_random_normal(rng, n)
        golden_μ = randn(rng, n)
        L = randn(rng, n, n)
        golden_precision = L * L' + Matrix{Float64}(I, n, n)
        return ExponentialFamily.MvGaussianMeanPrecision(golden_μ, golden_precision)
    end

    rng = StableRNG(42)
    
    n = 4

    mv_nm = naturalmanifold(
        ExponentialFamily.ExponentialFamilyDistribution{MvNormalMeanCovariance},
        ExponentialFamilyManifolds.NaturalParameterizationManifoldsOptions(n),
    )

    golden_dist = generate_random_normal(rng, n)

    point_on_nat_manifold = rand(rng, mv_nm.manifold)

    point2dist = ExponentialFamilyManifolds.point2dist(mv_nm, point_on_nat_manifold)

    samples = rand(rng, point2dist, 100)

    nat_logpdf = logpdf(point2dist, samples)

    cvi = CVI(rng, nothing, nothing, 100, nothing, ForwardDiffGrad(), 10, Val(false), false)

    ExpectationApproximations.cvi_objective(cvi, mv_nm, point_on_nat_manifold, golden_dist)
    
end

# struct CVI{R, M, AGB, O, G, B} <: AbstractApproximationMethod
#     rng::R
#     method::M
#     autograd_backend::AGB
#     n_iterations::Int
#     opt::O
#     grad::G
#     n_gradpoints::Int
#     enforce_proper_messages::Val{B}
#     warn::Bool
# end


# @testitem "cvi Bernoulli x Bernoulli prod" begin
#     include("./hybrid_setuptests.jl")
#     cvi_method = CVI(StableRNG(42), 500, Optimisers.Descent(0.01), ForwardDiffGrad(), 1000, Val(true), false)

#     for i in -1:1, j in -1:1
#         positive_measure = ExponentialFamily.ExponentialFamilyDistribution(Bernoulli, [i + 1e-12])
#         ef = ExponentialFamily.ExponentialFamilyDistribution(Bernoulli, [j + 1e-12])
#         product = ExpectationApproximations.approximate_distribution(cvi_method, CVIFactor(ef, positive_measure))
#         dist = convert(Distribution, product)
#         analytical_product = ExponentialFamily.ExponentialFamilyDistribution(Bernoulli, [i + j])
#         @test mean(dist) ≈ mean(analytical_product) atol = 0.01
#     end
# end

# @testitem "cvi `prod` tests" begin
#     include("./hybrid_setuptests.jl")
#     rng = StableRNG(42)

#     tests = ((method = CVI(StableRNG(42), 500, Optimisers.Descent(0.01), ForwardDiffGrad(), 2000, Val(true), false), tol = 0.1),)

#     # Check several prods against their analytical solutions
#     for test in tests, i in 1:5
#         # Univariate `Gamma`
#         g1 = GammaShapeRate(rand(rng) + 1, rand(rng) + 1)
#         g2 = GammaShapeRate(rand(rng) + 1, rand(rng) + 1)

#         cvi_factor = CVIFactor(convert(ExponentialFamilyDistribution, g1), convert(ExponentialFamilyDistribution, g2))


#         product = ExpectationApproximations.approximate_distribution(test.method, cvi_factor)
#         analytical_product = prod(ClosedProd(), g1, g2)

#         @test mean(product) ≈ mean(analytical_product) atol = test.tol
#     end
# end

# @testitem "Normal x Normal" begin
#     include("./hybrid_setuptests.jl")
#     seed = 123
#     rng = StableRNG(seed)
#     optimizer = Optimisers.Adam(0.01)
#     method = CVI(rng, 1000, optimizer, ForwardDiffGrad(), 2000, Val(false), false)

#     for i in -5:5
#         m_out, m_in = NormalMeanVariance(i, 1), NormalMeanVariance(0, 1)
#         cvi_factor = CVIFactor(convert(ExponentialFamilyDistribution, m_out), convert(ExponentialFamilyDistribution, m_in))

#         product = ExpectationApproximations.approximate_distribution(method, cvi_factor)
#         @test mean(product) ≈ mean(NormalWeightedMeanPrecision(i, 2.0)) atol = 0.1
#     end
# end

@testitem "MvNormal x MvNormal" begin
    include("./hybrid_setuptests.jl")
    rng = StableRNG(123)
    autograd = "finite", ManifoldDiff.FiniteDifferencesBackend()
    r_backend = ManifoldDiff.TangentDiffBackend(autograd)
    method = CVI(rng, r_backend, 100, ForwardDiffGrad(), 10, Val(false), false)
end