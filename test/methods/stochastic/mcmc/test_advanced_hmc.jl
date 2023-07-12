module TestAdvancedHMC

using Test
using ExpectationApproximations
using ExponentialFamily
using Distributions
using Random
using StableRNGs
using Optimisers
using AdvancedHMC
using ForwardDiff, ReverseDiff, LinearAlgebra
import Distributions: Univariate, Multivariate, Matrixvariate
import SpecialFunctions: digamma
import ExponentialFamily: KnownExponentialFamilyDistribution,convert

import ExpectationApproximations: AHMC, ImportanceSampling

@testset "AHMC" begin
    @testset "Constructor AHMC method" begin

        autograd = ReverseDiff
        automatic_ϵ = false
        ϵ_step = 0.01
        ϵ_jitter = 0.01 + 1e-8randn()
        integrator = JitteredLeapfrog(ϵ_step, ϵ_jitter)
        trajectory_sampler = SliceTS
        termination_criterion =  ClassicNoUTurn()

        for dim =2:10
            ahmc_default_method = AHMC(Multivariate, dim)
            
            @test ahmc_default_method.n_samples == 1000
            @test ahmc_default_method.n_adapts == 0
            @test ahmc_default_method.metric.M⁻¹  == ones(dim)
            @test ahmc_default_method.autograd == ForwardDiff
            @test ahmc_default_method.automatic_ϵ == true
            @test ahmc_default_method.integrator == Leapfrog(0.01)
            @test ahmc_default_method.trajectory_sampler == MultinomialTS
            @test ahmc_default_method.termination_criterion == GeneralisedNoUTurn()

            n_samples = dim*10
            n_adapts  = dim*5
            L = rand(dim)
            M = L*L' + 1e-8*diageye(dim)
            metric = DenseEuclideanMetric(M)
            ahmc_method = AHMC(Multivariate, dim,n_samples,n_adapts,metric, autograd,automatic_ϵ,integrator,trajectory_sampler,termination_criterion)

            @test ahmc_method.n_samples == n_samples
            @test ahmc_method.n_adapts == n_adapts
            @test ahmc_method.metric.M⁻¹  == M 
            @test ahmc_method.autograd == ReverseDiff
            @test ahmc_method.automatic_ϵ == false
            @test ahmc_method.integrator == integrator
            @test ahmc_method.trajectory_sampler == trajectory_sampler
            @test ahmc_method.termination_criterion == termination_criterion
        end
    end

    @testset "AHMC" begin
        @testset "AHMC: simple expectation with respect to normal" begin 
            true_dist  = NormalMeanVariance(5.0, 6.0)
            brtol = 0.1
            method = AHMC(Univariate, 1)
            @test isapprox(approximate(method, true_dist, (x) -> 1), 1.0, rtol = brtol)
            tmp = approximate(method, true_dist, (x) -> x)
            @test isapprox(tmp, mean(true_dist), rtol = brtol)
            @test isapprox(approximate(method, true_dist, (x) -> (x - tmp)^2), var(true_dist), rtol = brtol)
        end

        @testset "AHMC: product of normal and gamma" begin 
            method = AHMC(Univariate, 1,5000,5000)
            ExpectationApproximations.sample_direction(::NormalMeanVariance,::Gamma) = ExpectationApproximations.SampleRight()
            grid = -10:0.0001:30
            grid_positive = 0.00001:0.0001:30
            means = [-6, -3, 0, 1, 3,  6]
            vars =[1, 2, 3, 4, 5]
            as = [2, 5, 10]
            bs = [2, 5, 10]
            for m in means,v in vars, a in as, b in bs
                dist1  = NormalMeanVariance(m, v)
                dist2  = Gamma(a,b)
                prod_sample_list = prod(method,dist1,dist2)
                prod_samples = ExpectationApproximations.get_samples(prod_sample_list)
                mean_prod_ahmc = mean(prod_samples)
                logmean_prod_ahmc = mean( log.(prod_samples) )
                pdf_prod = (x) -> pdf(dist1,x)*pdf(dist2,x)
                normalization = sum(pdf_prod.(grid))
                pdf_prod_norm = (x) -> pdf_prod(x)/normalization
                mean_grid = mapreduce(x -> x*pdf_prod_norm(x), +, grid)
                logmean_grid = mapreduce(x -> log(x)*pdf_prod_norm(x), +, grid_positive)
                @test mean_prod_ahmc ≈ mean_grid atol = 0.3
                @test logmean_prod_ahmc ≈ logmean_grid atol = 0.1
            
            end
        end

        @testset "AHMC: Basic integration with unnormalized distribution: Beta" begin
            method = AHMC(Univariate, 1,1000,0)
            alphas = 5, 6, 20
            betas  = 2, 10, 20
            for alpha in alphas, beta in betas
                unnormalized_dist  = x -> x > 0 ? x^(alpha - 1) * (1 - x)^(beta - 1) : 1e-12
                tmp = approximate(method, unnormalized_dist, (x) -> x)
                @test isapprox(tmp, alpha / (beta + alpha), rtol = 0.2)
                @test isapprox(approximate(method, unnormalized_dist, (x) -> log(x)), digamma(alpha) - digamma(alpha + beta), rtol = 0.5)
            end
        end

        @testset "AHMC: Expectation of a multivariate function with respect to multivariate normal" begin
            f(x) = exp(-tr(x*x'))
            irng = StableRNG(123)
            D = 5
            nsamples = 5000
            method = AHMC(Multivariate, D, nsamples, 0)
            m = rand(D)
            L = rand(D)
            P = L*L' + 1e-6diageye(D)
            dist = MvNormalMeanCovariance(m,P)
            expecatation_ahmc = approximate(method,dist, f)

            method_is = ImportanceSampling(irng, 5000, dist; resampling = false)
            expectation_is = approximate(method_is, dist, f)
            @test expecatation_ahmc ≈ expectation_is rtol=0.1
        end

    end
end

end
