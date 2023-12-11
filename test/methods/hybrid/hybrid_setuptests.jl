using ExpectationApproximations
using ExponentialFamily
using Distributions
using Random
using StableRNGs
using Optimisers
using ExponentialFamilyManifolds
using ManifoldDiff
using LinearAlgebra
using FiniteDifferences
using Manopt
using ForwardDiff
using Manifolds
using BayesBase



function generate_random_normal(rng, n)
    golden_μ = randn(rng, n)
    L = randn(rng, n, n)
    golden_precision = L * L' + Matrix{Float64}(I, n, n)
    return ExponentialFamily.MvGaussianMeanPrecision(golden_μ, golden_precision)
end

function cvi_run(n::Int)
    r_backend = Manifolds.RiemannianProjectionBackend(ManifoldDiff.ForwardDiffBackend())
    method =  Manopt.gradient_descent
    stopping_criterion = StopWhenGradientNormLess(1e-4)
    rng = StableRNG(42)
    cvi = CVI(rng, method, r_backend, 100, method, ForwardDiffGrad(), 100, Val(false), false, stopping_criterion)
    m_out, m_in = generate_random_normal(rng, n), generate_random_normal(rng, n)
    cvi_factor = CVIFactor(convert(ExponentialFamilyDistribution, m_out), m_in)
    @show @code_warntype ExpectationApproximations.approximate_distribution(cvi, cvi_factor)
    product = ExpectationApproximations.approximate_distribution(cvi, cvi_factor)
    prod_result = prod(BayesBase.ClosedProd(), m_out, m_in)
    ef_prod_result = convert(ExponentialFamilyDistribution, prod_result)
    return ExponentialFamily.getnaturalparameters(ef_prod_result), ExponentialFamily.getnaturalparameters(product) 
end

@show cvi_run(2)[1] - cvi_run(2)[2]