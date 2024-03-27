@testitem "Unscented approximation method" begin
    using Test, Distributions

    import ExpectationApproximations: Unscented, unscented_statistics, approximate_expectation, FromTo, UnscentedParameters

    @testset "Univariate `unscented_statistics`" begin
        id = 1
        od = (1, )
        T = FromTo(Univariate, Univariate)
        unscented_parameters = UnscentedParameters(2, 3, 4)
        @test all(@inferred(unscented_statistics(Unscented{T}(id, od), 0.0, 1.0,(x) -> x, Val(true))) .≈ (0.0, 1.0, 1.0))
        @test all(@inferred(unscented_statistics(Unscented{T}(id, od), 0.0, 1.0,(x) -> x^2, Val(true))) .≈ (1.0, 2.0, 0.0))
        @test all(@inferred(unscented_statistics(Unscented{T}(id, od), -3.0, 1.0, (x) -> x^2, Val(true))) .≈ (10.0, 38.0, -6.0))
        @test all(isapprox.(@inferred(unscented_statistics(Unscented{T}(id, od), 0.0, 1.0, (x) -> sin(x) + cos(x), Val(true))), (0.5, 1.5, 1.0); atol = 1e-6))
        @test all(isapprox.(@inferred(unscented_statistics(Unscented{T}(id, od), -3.0, 3.0, (x) -> sin(x) + cos(x), Val(true))), (0.56555582, 7.91911820, -2.54661619); atol = 1e-6))

        @test all(@inferred(unscented_statistics(Unscented{T}(id, od, unscented_parameters), 0.0, 3.0,(x) -> x, Val(true))) .≈ (0.0, 3.0, 3.0))
        @test all(@inferred(unscented_statistics(Unscented{T}(id, od, unscented_parameters), -1.0, 1.0 , (x) -> x^2, Val(true))) .≈ (2.0, 23.0, -2.0))
        @test all(@inferred(unscented_statistics(Unscented{T}(id, od, unscented_parameters), 0.0, 3.0, (x) -> x^2, Val(true))) .≈ (3.0, 171.0, 0.0))
        @test all(
            isapprox.(
                @inferred(unscented_statistics(Unscented{T}(id, od, unscented_parameters), 0.0, 1.0, (x) -> sin(x) + cos(x), Val(true))),
                (0.9381025804009704, 0.11996354864504076, -0.21718431835123952);
                atol = 1e-6
            )
        )
        @test all(
            isapprox.(
                @inferred(unscented_statistics(Unscented{T}(id, od, unscented_parameters), -3.0, 3.0, (x) -> sin(x) + cos(x), Val(true))),
                (-1.0806538600139532, 0.08398591776750242, -0.32685086597452473);
                atol = 1e-6
            )
        )
    end

    @testset "Multivariate `unscented_statistics`" begin
        id = 2
        od = (2, )
        T = FromTo(Multivariate, Multivariate)
        unscented_parameters = UnscentedParameters(2, 3, 4)
        unscented_parameters2 = UnscentedParameters(0.5, 0.1, 0.2)
        @test all(@inferred(unscented_statistics(Unscented{T}(id, od), [0.0, 0.0], [1.0 0.0; 0.0 1.0], (x) -> x, Val(true))) .≈ ([0.0, 0.0], [1.0 0.0; 0.0 1.0], [1.0 0.0; 0.0 1.0]))
        @test all(@inferred(unscented_statistics(Unscented{T}(id, od), [2.0, -3.0], [4.0 -1.0; -1.0 2.0], (x) -> x, Val(true))) .≈ ([2.0, -3.0], [4.0 -1.0; -1.0 2.0], [4.0 -1.0; -1.0 2.0]))

        @test all(
            isapprox.(@inferred(unscented_statistics(Unscented{T}(id, od), [1.0, -1.0], [1.0 0.0; 0.0 1.0],(x) -> x .^ 2, Val(true))), ([2.0, 2.0], [6.0 2.0; 2.0 6.0], [2.0 0; 0.0 -2.0]), atol = 1e-4)
        )
        @test all(
            isapprox.(
                @inferred(unscented_statistics(Unscented{T}(id, od), [2.0, -3.0], [4.0 -1.0; -1.0 2.0],(x) -> x .^ 2, Val(true))),
                ([8.0, 11.0], [96.0 40.0; 40.0 80.0], [16.0 6.0; -4.0 -12.0]),
                atol = 1e-4
            )
        )

        # Compared against the ForneyLab implementation
        @test all(
            @inferred(unscented_statistics(Unscented{T}(id, od, unscented_parameters),[0.0, 0.0],[1.0 0.0; 0.0 1.0],(x) -> x, Val(true))) .≈
            ([0.0, 0.0], [1.0 0.0; 0.0 1.0], [1.0 0.0; 0.0 1.0])
        )
        @test all(
            @inferred(unscented_statistics(Unscented{T}(id, od, unscented_parameters), [2.0, -3.0],[4.0 -1.0; -1.0 2.0],(x) -> x, Val(true))) .≈
            ([2.0, -3.0], [4.0 -1.0; -1.0 2.0], [4.0 -1.0; -1.0 2.0])
        )

        @test all(
            isapprox.(
                @inferred(unscented_statistics(Unscented{T}(id, od, unscented_parameters), [1.0, -1.0], [1.0 0.0; 0.0 1.0], (x) -> x .^ 2, Val(true) )),
                ([2.0, 2.0], [27.0 -1.0; -1.0 27.0], [2.0 0; 0.0 -2.0]),
                atol = 1e-4
            )
        )
        @test all(
            isapprox.(
                @inferred(unscented_statistics(Unscented{T}(id, od, unscented_parameters2),[2.0, -3.0],[4.0 -1.0; -1.0 2.0], (x) -> x .+ 1, Val(true))),
                ([3.0, -2.0], [4.0 -1.0; -1.0 2.0], [4.0 -1.0; -1.0 2.0]),
                atol = 1e-4
            )
        )
    end

    @testset "Unscented: Basic integrations/approximations with Gaussian p" begin
        true_dist = Normal(5.0, 6.0)
        sigmas    = [5, 6, 7, 8, 9, 10, 11, 16]
        means     = [3, 4, 7, 8]
        brtol     = 0.1
        unscented_parameters = UnscentedParameters(0.5,0.1,0.2)
        for sigma in sigmas, m in means
            importance_dist = Normal(m, sigma)
            method = @inferred(Unscented{FromTo(Univariate, Univariate)}(1, 1, unscented_parameters))
            @test isapprox(@inferred(approximate_expectation(method, true_dist, (x) -> 1)), 1.0, rtol = brtol)
            tmp = @inferred(approximate_expectation(method, true_dist, (x) -> x))
            @test isapprox(tmp, mean(true_dist), rtol = brtol)
            @test isapprox(@inferred(approximate_expectation(method, true_dist, (x) -> (x - tmp)^2)), var(true_dist), rtol = brtol)
        end
    end
end
