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
            ahmc_method = AHMC(dim,metric, n_samples,n_adapts,autograd,automatic_ϵ,integrator,trajectory_sampler,termination_criterion)

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

end

end
