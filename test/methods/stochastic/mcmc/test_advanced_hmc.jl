module TestAdvancedHMC

using Test
using ExpectationApproximations
using ExponentialFamily
using Distributions
using Random
using StableRNGs
using Optimisers
using AdvancedHMC
using ForwardDiff, ReverseDiff

import ExpectationApproximations: AHMC

@testset "AHMC" begin
    @testset "Constructor AHMC method" begin

        autograd = ReverseDiff
        automatic_ϵ = false
        ϵ_step = 0.01
        ϵ_jitter = 0.01 + 1e-8randn()
        integrator = JitteredLeapfrog(ϵ_step, ϵ_jitter)
        trajectory_sampler = SliceTS
        termination_criterion =  ClassicNoUTurn()

        for dim =1:10
            ahmc_default_method = AHMC(dim)
            
            @test ahmc_default_method.n_samples == 1000
            @test ahmc_default_method.n_adapts == 1000
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
            ahmc_method = AHMC(dim,n_samples,n_adapts,metric, autograd,automatic_ϵ,integrator,trajectory_sampler,termination_criterion)

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

    @testset "ahmc" begin
        @testset "ahmc: expectation with respect to normal" begin 
            true_dist  = NormalMeanVariance(5.0, 6.0)
            brtol = 0.1
            method = AHMC(1)
            @test isapprox(approximate(method, true_dist, (x) -> 1), 1.0, rtol = brtol)
            tmp = approximate(method, true_dist, (x) -> x)
            @test isapprox(tmp, mean(true_dist), rtol = brtol)
            @test isapprox(approximate(method, true_dist, (x) -> (x - tmp)^2), var(true_dist), rtol = brtol)
        end
    end

end

end
