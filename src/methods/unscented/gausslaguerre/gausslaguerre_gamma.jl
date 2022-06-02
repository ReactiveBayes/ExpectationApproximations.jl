using Distributions

# θ = 1/β
# logg(x) = logf(x/β) + (α - 1) * log(x) - loggamma(α)
adjust_logintegrand(::GaussLaguerre, q::Gamma, logf::F) where F = let α = shape(q), θ = scale(q)
    (x) -> logf(θ * x) + (α - 1) * log(x) # We subtract `loggamma(α)` later on, see `adjust_logresult`
end

# θ = 1/β
# g(x) = f(x/β) * x^(α - 1) / Γ(x)
adjust_integrand(::GaussLaguerre, q::Gamma, f::F) where F = let α = shape(q), θ = scale(q)
    (x) -> f(θ * x) * x ^ (α - 1) # we divide by Γ(α) later on, see `adjust_result`
end
    
adjust_logresult(::GaussLaguerre, q::Gamma, logresult) = logresult - loggamma(shape(q))

# Computing `gamma(α)` prone to overflowing
adjust_result(::GaussLaguerre, q::Gamma, result) = real(exp(log(Complex(result)) - loggamma(shape(q))))