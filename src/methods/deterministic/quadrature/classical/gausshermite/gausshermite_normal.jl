using Distributions

# g(x) = f(x * sqrt(2) * σ + μ)
function adjust_integrand(::GaussHermite, q::Normal, f::F) where {F}
    return let μ = mean(q), sqrt2σ = sqrt2 * std(q)
        (x) -> f(sqrt2σ * x + μ) # we divide by `sqrt(π)` later on, see `adjust_result`
    end
end

# logg(x) = logf(x * sqrt(2) * σ + μ)
function adjust_integrand(::GaussHermite, q::Normal, f::LogIntegrand)
    return let μ = mean(q), sqrt2σ = sqrt2 * std(q), logf = log(f)
        LogIntegrand((x) -> logf(sqrt2σ * x + μ)) # We subtract `log(sqrt(π))` later on, see `adjust_result`
    end
end

adjust_result(::GaussHermite, q::Normal, result)            = invsqrtπ * result
adjust_result(::GaussHermite, q::Normal, result::LogResult) = LogResult(log(result) - logsqrtπ)
