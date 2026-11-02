@testsnippet AHMCSetup begin
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
    import ExponentialFamily: ExponentialFamilyDistribution, convert
    import ExpectationApproximations: AHMC, ImportanceSampling

    include("./../../../diageye_setup.jl")
end
