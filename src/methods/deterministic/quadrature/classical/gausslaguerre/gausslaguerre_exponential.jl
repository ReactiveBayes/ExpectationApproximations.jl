using Distributions

# g(x) = f(x/λ)
function adjust_integrand(::GaussLaguerre, q::Exponential, f::F) where {F}
    return let invλ = inv(rate(q))
        (x) -> f(invλ * x)
    end
end

# logg(x) = logf(x/λ)
function adjust_integrand(::GaussLaguerre, q::Exponential, f::LogIntegrand)
    return let invλ = inv(rate(q)), logf = log(f)
        LogIntegrand((x) -> logf(invλ * x))
    end
end

adjust_result(::GaussLaguerre, q::Exponential, result) = result
