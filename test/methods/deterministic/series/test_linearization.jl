module ExpectationApproximationsLinearizationTest

using Test
using ExpectationApproximations
import ExpectationApproximations: Linearization, approximate

@testset "Linearization approximation method" begin
    @testset "linearization `approximate` tests" begin
        @test @inferred(approximate(Linearization(), (x, y) -> x + y, (1, 2))) == ([1 1], 0)
        @test @inferred(approximate(Linearization(), (x, y) -> x - y, (1, 2))) == ([1 -1], 0)
        @test @inferred(approximate(Linearization(), (x, y) -> x .- y, ([1.0, 2.0], 1.0))) == ([1.0 0.0 -1.0; 0.0 1.0 -1.0], [0.0, 0.0])
        @test @inferred(approximate(Linearization(), (x, y) -> x .- y, ([1.0, 2.0], [1.0, 1.0]))) == ([1.0 0.0 -1.0 0.0; 0.0 1.0 0.0 -1.0], [0.0, 0.0])
    end
end

end
