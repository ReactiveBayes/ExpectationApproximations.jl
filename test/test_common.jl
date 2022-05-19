module TestsCommon

    using Test

    @testset "Points wrapper" begin
        import ExpectationApproximations: Points
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
        import ExpectationApproximations: Weights
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

    @testset "ScaleArguments" begin 
        import ExpectationApproximations: RescaleArguments

        for ucallable in (sin, cos, abs2, (x) -> x + 1), scale in (1, 2.0, 0.5)
            for rescaled in (RescaleArguments(ucallable, scale), RescaleArguments(ucallable, (scale, )))
                for value in rand(eltype(scale), 5)
                    @test ucallable(scale * value) === rescaled(value)
                end
            end
        end

        for ubcallable in (+, -, *), scale in (1, 2.0, 0.5, (3.0, 2.0))
            rescaled = RescaleArguments(ubcallable, scale)
            for left in rand(3), right in rand(3)
                @test ubcallable((scale .* (left, right))...) === rescaled(left, right)
            end
        end

        for n in (2, 3), mcallable in (sum, (x) -> x .+ 1), scale in (1, 2.0, 0.5)
            for rescaled in (RescaleArguments(mcallable, scale), RescaleArguments(mcallable, (scale, )))
                for value in map(_ -> rand(eltype(scale), n), 1:5)
                    @test mcallable(scale .* value) == rescaled(value)
                end
            end
        end

        for mbcallable in ((x, y) -> sum(x) + sum(y), ), scale in (1, 2.0, 0.5, (-1.0, 1.0))
            rescaled = RescaleArguments(mbcallable, scale)
            for left in map(_ -> rand(eltype(scale), 2), 1:3), right in map(_ -> rand(eltype(scale), 2), 1:3)
                @test mbcallable((scale .* (left, right))...) == rescaled(left, right)
            end
        end

        @test_throws AssertionError RescaleArguments((x, y) -> x + y, (1.0, 2.0, 3.0))(1.0, 2.0)
        @test_throws AssertionError RescaleArguments((x, y) -> x + y, (1.0, ))(1.0, 2.0)
    end

end