module TestsTransforms

    using Test
    using ExpectationApproximations

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

    @testset "ShiftOutput" begin 
        import ExpectationApproximations: ShiftOutput

        for ucallable in (sin, cos, abs2, (x) -> x + 1), shift in (1, -2.0, 0.5)
            for shifted in (ShiftOutput(ucallable, shift), )
                for value in rand(eltype(shift), 5)
                    @test ucallable(value) + shift === shifted(value)
                end
            end
        end

        for ubcallable in (+, -, *), shift in (1, -2.0, 0.5)
            shifted = ShiftOutput(ubcallable, shift)
            for left in rand(3), right in rand(3)
                @test ubcallable(left, right) + shift === shifted(left, right)
            end
        end

        for n in (2, 3), mcallable in ((x) -> x .+ 1, ), shift in (1, -2.0, 0.5)
            for shifted in (ShiftOutput(mcallable, shift), ShiftOutput(mcallable, fill(shift, n)))
                for value in map(_ -> rand(eltype(shift), n), 1:5)
                    @test mcallable(value) .+ shift == shifted(value)
                end
            end
        end

        for mbcallable in ((x, y) -> sum(x) + sum(y), ), shift in (1, 2.0, 0.5, )
            shifted = ShiftOutput(mbcallable, shift)
            for left in map(_ -> rand(eltype(shift), 2), 1:3), right in map(_ -> rand(eltype(shift), 2), 1:3)
                @test mbcallable(left, right) .+ shift == shifted(left, right)
            end
        end
    end

    @testset "Decay" begin 
        import ExpectationApproximations: Decay

        for ucallable in (sin, cos, abs2, (x) -> x + 1)
            rescaled = Decay(ucallable)
            for value in rand(Float64, 5)
                @test ucallable(value) / value ≈ rescaled(value)  
            end
        end

        for ucallable in (sin, cos, abs2, (x) -> x + 1), decay in (1, 2.0, 0.5)
            for rescaled in (Decay(ucallable, decay), )
                for value in rand(eltype(decay), 5)
                    @test ucallable(value) / (value^decay) ≈ rescaled(value)
                end
            end
        end

    end

    @testset "Displace" begin 
        import ExpectationApproximations: Displace

        for ucallable in (sin, cos, abs2, (x) -> x + 1)
            displaced = Displace(ucallable)
            for value in rand(Float64, 5)
                @test ucallable(value) + value ≈ displaced(value)  
            end
        end

        for ucallable in (sin, cos, abs2, (x) -> x + 1), scale in (1, 2.0, 0.5)
            for displaced in (Displace(ucallable, scale), )
                for value in rand(eltype(scale), 5)
                    @test ucallable(value) + scale * value ≈ displaced(value)
                end
            end
        end

    end
    

end