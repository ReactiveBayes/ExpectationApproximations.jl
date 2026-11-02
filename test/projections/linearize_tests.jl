@testitem "linearize: the first-order expansion in every shape" tags = [:projections] setup = [EAHelpers] begin
    using ForwardDiff
    # scalar to scalar
    A, b = EA.linearize(sin, 0.3)
    @test A ≈ cos(0.3)
    @test b ≈ sin(0.3) - cos(0.3) * 0.3
    # vector to vector
    f(x) = [x[1] * x[2], x[1] + sin(x[2])]
    x = [0.5, -1.2]
    A, b = EA.linearize(f, x)
    @test A ≈ ForwardDiff.jacobian(f, x)
    @test A * x + b ≈ f(x)
    # vector to scalar: a row
    A, b = EA.linearize(x -> x[1]^2 + x[2], x)
    @test A ≈ [1.0 1.0]
    @test A * x + b ≈ x[1]^2 + x[2]
    # scalar to vector: a column
    A, b = EA.linearize(t -> [t, t^2], 2.0)
    @test A ≈ [1.0, 4.0]
    @test A * 2.0 + b ≈ [2.0, 4.0]
end

@testitem "linearize: several inputs, jointly in their concatenation" tags = [:projections] setup = [EAHelpers] begin
    # f(x, y) = x y₁ + y₂ at x = 2, y = (3, 4): the gradient over (x, y₁, y₂) is (3, 2, 1)
    A, b = EA.linearize((x, y) -> x * y[1] + y[2], 2.0, [3.0, 4.0])
    @test A ≈ [3.0 2.0 1.0]
    @test A * [2.0, 3.0, 4.0] + b ≈ 10.0
    # a linear function is reproduced exactly
    M, c = [1.0 2.0 0.0; 0.0 1.0 3.0], [0.5, -1.0]
    A, b = EA.linearize((x, y) -> M * [x; y] + c, 1.0, [2.0, 3.0])
    @test A ≈ M
    @test b ≈ c
end
