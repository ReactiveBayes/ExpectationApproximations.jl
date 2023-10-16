@testitem "SampleList" begin
    using Test
    using ExpectationApproximations
    using Random
    using Distributions
    using LinearAlgebra
    using StaticArrays
    using StableRNGs

    import ExpectationApproximations: SampleList, vague
    import ExpectationApproximations: deep_eltype, get_samples, get_weights, sample_list_zero_element
    import ExpectationApproximations: variate_form, Univariate, Multivariate, Matrixvariate
    import ExpectationApproximations: get_meta, is_meta_present, get_unnormalised_weights, get_entropy, get_logproposal, get_logintegrand
    import ExpectationApproximations: call_logproposal, call_logintegrand
    import ExpectationApproximations: transform_samples, transform_weights!
    import ExpectationApproximations: xtlog, mirrorlog

    @testset "Internal functions" begin
        @test sample_list_zero_element(SampleList([1.0, 1.0])) === 0.0
        @test sample_list_zero_element(SampleList([[1.0, 1.0], [1.0, 1.0]])) == [0.0, 0.0]
        @test sample_list_zero_element(SampleList([[1.0 1.0; 1.0 1.0], [1.0 1.0; 1.0 1.0]])) == [0.0 0.0; 0.0 0.0]
        @test sample_list_zero_element(SampleList([[1.0 1.0 1.0; 1.0 1.0 1.0], [1.0 1.0 1.0; 1.0 1.0 1.0]])) == [0.0 0.0 0.0; 0.0 0.0 0.0]
        @test sample_list_zero_element(SampleList([[1.0; 1.0], [1.0; 1.0]])) == [0.0; 0.0]
    end

    @testset "Constructor" begin
        rng = StableRNG(1234)

        for N in [5, 10, 100], type in [Float64, Float32, BigFloat]
            scalar_samples    = rand(rng, type, N)
            scalar_samplelist = SampleList(scalar_samples)

            @test collect(get_samples(scalar_samplelist)) == first.(collect(scalar_samplelist))
            @test collect(get_weights(scalar_samplelist)) == fill(one(type) / N, N)
            @test deep_eltype(scalar_samplelist) === type
            @test eltype(scalar_samplelist) === Tuple{type, type}
            @test eltype(get_weights(scalar_samplelist)) === type
            @test variate_form(scalar_samplelist) === Univariate
            @test is_meta_present(scalar_samplelist) === false

            scalar_weights = rand(rng, type, N)

            @test_throws AssertionError SampleList(scalar_samples, scalar_weights)
            scalar_weights ./= sum(scalar_weights)
            scalar_samplelist = SampleList(scalar_samples, scalar_weights)

            @test collect(get_samples(scalar_samplelist)) == first.(collect(scalar_samplelist))
            @test collect(get_weights(scalar_samplelist)) == scalar_weights
            @test deep_eltype(scalar_samplelist) === type
            @test eltype(scalar_samplelist) === Tuple{type, type}
            @test eltype(get_weights(scalar_samplelist)) === type
            @test variate_form(scalar_samplelist) === Univariate
            @test is_meta_present(scalar_samplelist) === false

            vector_samples    = [rand(rng, type, 2) for _ in 1:N]
            vector_samplelist = SampleList(vector_samples)

            @test collect(get_samples(vector_samplelist)) == first.(collect(vector_samplelist))
            @test collect(get_weights(vector_samplelist)) == fill(one(type) / N, N)
            @test deep_eltype(vector_samplelist) === type
            @test eltype(vector_samplelist) === Tuple{SVector{2, type}, type}
            @test eltype(get_weights(vector_samplelist)) === type
            @test variate_form(vector_samplelist) === Multivariate
            @test is_meta_present(vector_samplelist) === false

            vector_weights = rand(rng, type, N)
            @test_throws AssertionError SampleList(vector_samples, vector_weights)
            vector_weights ./= sum(vector_weights)
            vector_samplelist = SampleList(vector_samples, vector_weights)

            @test collect(get_samples(vector_samplelist)) == first.(collect(vector_samplelist))
            @test collect(get_weights(vector_samplelist)) == vector_weights
            @test deep_eltype(vector_samplelist) === type
            @test eltype(vector_samplelist) === Tuple{SVector{2, type}, type}
            @test eltype(get_weights(vector_samplelist)) === type
            @test variate_form(vector_samplelist) === Multivariate
            @test is_meta_present(vector_samplelist) === false

            matrix_samples    = [rand(rng, type, 2, 2) for _ in 1:N]
            matrix_samplelist = SampleList(matrix_samples)

            @test collect(get_samples(matrix_samplelist)) == first.(collect(matrix_samplelist))
            @test collect(get_weights(matrix_samplelist)) == fill(one(type) / N, N)
            @test deep_eltype(matrix_samplelist) === type
            @test eltype(matrix_samplelist) === Tuple{SMatrix{2, 2, type, 4}, type}
            @test eltype(get_weights(matrix_samplelist)) === type
            @test variate_form(matrix_samplelist) === Matrixvariate
            @test is_meta_present(matrix_samplelist) === false

            matrix_weights = rand(rng, type, N)
            @test_throws AssertionError SampleList(matrix_samples, matrix_weights)
            matrix_weights ./= sum(matrix_weights)
            matrix_samplelist = SampleList(matrix_samples, matrix_weights)

            @test collect(get_samples(matrix_samplelist)) == first.(collect(matrix_samplelist))
            @test collect(get_weights(matrix_samplelist)) == matrix_weights
            @test deep_eltype(matrix_samplelist) === type
            @test eltype(matrix_samplelist) === Tuple{SMatrix{2, 2, type, 4}, type}
            @test eltype(get_weights(matrix_samplelist)) === type
            @test variate_form(matrix_samplelist) === Matrixvariate
            @test is_meta_present(matrix_samplelist) === false
        end

        @test_throws AssertionError SampleList(rand(10), rand(5))
        @test_throws AssertionError SampleList(rand(5), rand(10))
        @test_throws AssertionError SampleList(rand(5), [-1 for _ in 1:5])

        @test_throws AssertionError SampleList([rand(10) for _ in 1:10], rand(5))
        @test_throws AssertionError SampleList([rand(5) for _ in 1:5], rand(10))
        @test_throws AssertionError SampleList([rand(5) for _ in 1:5], [-1 for _ in 1:5])

        @test_throws AssertionError SampleList([rand(10, 10) for _ in 1:10], rand(5))
        @test_throws AssertionError SampleList([rand(5, 5) for _ in 1:5], rand(10))
        @test_throws AssertionError SampleList([rand(5, 5) for _ in 1:5], [-1 for _ in 1:5])
    end
end
