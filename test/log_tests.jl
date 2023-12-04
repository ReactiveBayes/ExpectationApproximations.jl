@testitem "LogIntegrand wrapper" begin
    import LogarithmicNumbers: Logarithmic

    testfn = ((x) -> -(x - 1.0)^2, (x) -> exp(Logarithmic, -x))

    vals = (1.0, 2.0, 3.0)

    for f in testfn, v in vals
        logf = LogIntegrand(f)

        @test logf(v) == exp(f(v))
        @test log(logf)(v) == f(v)

        logf = convert(LogIntegrand, LogIntegrand(f))

        @test logf(v) == exp(f(v))
        @test log(logf)(v) == f(v)

        logf = convert(LogIntegrand, (x) -> exp(f(x)))

        @test real(logf(v)) ≈ exp(f(v))
        @test imag(logf(v)) ≈ 0.0

        @test real(log(logf)(v)) ≈ float(f(v))
        @test imag(log(logf)(v)) ≈ 0.0
    end
end