module TestsCommon

    using Test
    using ExpectationApproximations

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

    @testset "Pdf wrapper" begin 
        import DomainSets: HalfLine

        f   = (x) -> exp(-(x - 1.0)^2)
        pdf = Pdf(f)

        @test pdf(1.0) == f(1.0)
        @test pdf(2.0) == f(2.0)

        f   = (x) -> exp(-(x - 1.0)^2)
        pdf = Pdf(HalfLine(), f)

        @test pdf(1.0) == f(1.0)
        @test pdf(2.0) == f(2.0)
        @test_throws AssertionError pdf(-1.0)
        @test_throws AssertionError pdf(-2.0)

        f   = (x) -> exp(-(x .- 1.0)' * (x .- 1.0))
        pdf = Pdf(f)

        @test pdf(ones(2)) == f(ones(2))
        @test pdf(zeros(2)) == f(zeros(2))
        
        f   = (x) -> exp(-(x .- 1.0)' * (x .- 1.0))
        pdf = Pdf(HalfLine() ^ 2, f)

        @test pdf(ones(2)) == f(ones(2))
        @test pdf(zeros(2)) == f(zeros(2))
        @test_throws AssertionError pdf([ -1.0, 1.0  ])
        @test_throws AssertionError pdf([ 1.0, -1.0  ])
        @test_throws AssertionError pdf([ -1.0, -1.0  ]) 
    end

    @testset "Pdf wrapper" begin 
        import DomainSets: HalfLine

        f      = (x) -> -(x - 1.0)^2
        logpdf = LogPdf(f)

        @test logpdf(1.0) == f(1.0)
        @test logpdf(2.0) == f(2.0)

        f      = (x) -> -(x - 1.0)^2
        logpdf = LogPdf(HalfLine(), f)

        @test logpdf(1.0) == f(1.0)
        @test logpdf(2.0) == f(2.0)
        @test_throws AssertionError logpdf(-1.0)
        @test_throws AssertionError logpdf(-2.0)

        f      = (x) -> -(x .- 1.0)' * (x .- 1.0)
        logpdf = LogPdf(f)

        @test logpdf(ones(2)) == f(ones(2))
        @test logpdf(zeros(2)) == f(zeros(2))
        
        f      = (x) -> -(x .- 1.0)' * (x .- 1.0)
        logpdf = LogPdf(HalfLine() ^ 2, f)

        @test logpdf(ones(2)) == f(ones(2))
        @test logpdf(zeros(2)) == f(zeros(2))
        @test_throws AssertionError logpdf([ -1.0, 1.0  ])
        @test_throws AssertionError logpdf([ 1.0, -1.0  ])
        @test_throws AssertionError logpdf([ -1.0, -1.0  ]) 
    end

    @testset "RescaleArguments" begin 
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

    @testset "RescaleOutput" begin 
        import ExpectationApproximations: RescaleOutput

        for ucallable in (sin, cos, abs2, (x) -> x + 1), scale in (1, 2.0, 0.5)
            for rescaled in (RescaleOutput(ucallable, scale), )
                for value in rand(eltype(scale), 5)
                    @test scale * ucallable(value) === rescaled(value)
                end
            end
        end

        for ubcallable in (+, -, *), scale in (1, 2.0, 0.5)
            rescaled = RescaleOutput(ubcallable, scale)
            for left in rand(3), right in rand(3)
                @test scale * ubcallable(left, right) === rescaled(left, right)
            end
        end

        for n in (2, 3), mcallable in ((x) -> x .+ 1, ), scale in (1, 2.0, 0.5)
            for rescaled in (RescaleOutput(mcallable, scale), RescaleOutput(mcallable, fill(scale, n)))
                for value in map(_ -> rand(eltype(scale), n), 1:5)
                    @test scale .* mcallable(value) == rescaled(value)
                end
            end
        end

        for mbcallable in ((x, y) -> sum(x) + sum(y), ), scale in (1, 2.0, 0.5, )
            rescaled = RescaleOutput(mbcallable, scale)
            for left in map(_ -> rand(eltype(scale), 2), 1:3), right in map(_ -> rand(eltype(scale), 2), 1:3)
                @test scale .* mbcallable(left, right) == rescaled(left, right)
            end
        end
    end

end