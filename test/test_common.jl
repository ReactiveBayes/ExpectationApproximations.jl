module TestsCommon

    using Test

    @testset "Points wrapper" begin
        import ExpectationApproximations.Points
        import Base: iterate, IteratorSize, IteratorEltype, eltype, length, size

        for n in (10, 20), T in (Int, Float64)
            parent = rand(T, n)
            points = Points(parent)

            @test eltype(points) === eltype(parent)
            @test eltype(points) === T
            @test IteratorSize(points) === IteratorSize(parent)
            @test IteratorSize(typeof(points)) === IteratorSize(typeof(parent))
            @test IteratorEltype(points) === IteratorEltype(parent)
            @test IteratorEltype(typeof(points)) === IteratorEltype(typeof(parent))
            @test eltype(points) === eltype(parent)
            @test eltype(typeof(points)) === eltype(typeof(parent))
            @test length(points) === length(parent)

            @test all((d) -> d[1] === d[2], zip(parent, points))
            @test all((d) -> d[1] === d[2], zip(map(abs2, parent), map(abs2, points)))
        end
    end

    @testset "Weights wrapper" begin
        import ExpectationApproximations.Weights
        import Base: iterate, IteratorSize, IteratorEltype, eltype, length, size

        for n in (10, 20), T in (Int, Float64)
            parent = rand(T, n)
            points = Weights(parent)

            @test eltype(points) === eltype(parent)
            @test eltype(points) === T
            @test IteratorSize(points) === IteratorSize(parent)
            @test IteratorSize(typeof(points)) === IteratorSize(typeof(parent))
            @test IteratorEltype(points) === IteratorEltype(parent)
            @test IteratorEltype(typeof(points)) === IteratorEltype(typeof(parent))
            @test eltype(points) === eltype(parent)
            @test eltype(typeof(points)) === eltype(typeof(parent))
            @test length(points) === length(parent)

            @test all((d) -> d[1] === d[2], zip(parent, points))
            @test all((d) -> d[1] === d[2], zip(map(abs2, parent), map(abs2, points)))
        end
    end

end