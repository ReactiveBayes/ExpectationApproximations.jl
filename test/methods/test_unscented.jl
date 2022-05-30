module TestsUnscented

    using Test
    using ExpectationApproximations
    using Distributions

    @testset "GaussLaguerre" begin

        @testset "LogFn-version" begin 
            import ExpectationApproximations: mean_var

            for α1 in (1.3, 2.1, 10.1), 
                α2 in (1.3, 2.1, 10.1),
                β1 in (1.3, 2.1, 10.1),
                β2 in (1.3, 2.1, 10.1),
                n in (61, 121, )

                d1 = Gamma(α1, β1)
                d2 = Gamma(α2, β2)

                pα = α1 + α2 - 1
                pβ = (β1 * β2) / (β1 + β2)
                p  = Gamma(pα, pβ)

                m, v = mean_var(Unscented(GaussLaguerre(n)), d1, LogFn((x) -> logpdf(d2, x)))
                
                @test isapprox(m, mean(p), rtol = 1e-1)
                @test isapprox(v, var(p), rtol = 1e-1)

            end

        end

    end

end