using Distributions

# logg(x) = logf(x/λ)
adjust_logintegrand(::GaussLaguerre, q::Exponential, logf::F) where F = let invλ = inv(rate(q))
    (x) -> logf(invλ * x)
end

# g(x) = f(x/λ)
adjust_integrand(::GaussLaguerre, q::Exponential, f::F) where F = let invλ = inv(rate(q))
    (x) -> f(invλ * x)
end
    
adjust_logresult(::GaussLaguerre, q::Exponential, logresult) = logresult
adjust_result(::GaussLaguerre, q::Exponential, result)       = result