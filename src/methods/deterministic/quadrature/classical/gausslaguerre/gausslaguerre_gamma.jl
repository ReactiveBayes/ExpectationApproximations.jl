using Distributions

# θ = 1/β
# g(x) = f(x/β) * x^(α - 1) / Γ(x)
function adjust_integrand(::GaussLaguerre, q::Gamma, f::F) where {F}
    return let α = shape(q), θ = scale(q)
        (x) -> f(θ * x) * x^(α - 1) # we divide by Γ(α) later on, see `adjust_result`
    end
end

# θ = 1/β
# logg(x) = logf(x/β) + (α - 1) * log(x) - loggamma(α)
function adjust_integrand(::GaussLaguerre, q::Gamma, f::LogIntegrand)
    return let α = shape(q), θ = scale(q), logf = log(f)
        LogIntegrand((x) -> logf(θ * x) + (α - 1) * log(x)) # We subtract `loggamma(α)` later on, see `adjust_result`
    end
end

# Computing `gamma(α)` prone to overflowing
adjust_result(::GaussLaguerre, q::Gamma, result) = convert(Logarithmic, LogResult(log(Complex(result)) - loggamma(shape(q))))
adjust_result(::GaussLaguerre, q::Gamma, result::LogResult) = LogResult(log(result) - loggamma(shape(q)))
