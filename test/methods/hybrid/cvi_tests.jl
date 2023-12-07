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

    cvi = CVI(rng, nothing, nothing, 100, nothing, ForwardDiffGrad(), 10, Val(false), false)

    ExpectationApproximations.cvi_objective(cvi, mv_nm, point_on_nat_manifold, golden_dist)
    
end

@testitem "MvNormal x MvNormal" begin
    include("./hybrid_setuptests.jl")
    rng = StableRNG(123)
    autograd = ManifoldDiff.FiniteDifferencesBackend()
    r_backend = ManifoldDiff.TangentDiffBackend(autograd)
    method = CVI(rng, nothing, r_backend, 100, nothing, ForwardDiffGrad(), 10, Val(false), false)
    for n in 2:4
        m_out, m_in = generate_random_normal(rng, n), generate_random_normal(rng, n)
        cvi_factor = CVIFactor(convert(ExponentialFamilyDistribution, m_out), convert(ExponentialFamilyDistribution, m_in))
        product = ExpectationApproximations.approximate_distribution(method, cvi_factor)
        @test mean(product) ≈ mean(NormalWeightedMeanPrecision(i, 2.0)) atol = 0.1
    end
end