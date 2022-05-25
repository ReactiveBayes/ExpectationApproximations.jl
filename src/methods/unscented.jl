export Unscented, GaussLaguerre

using FastGaussQuadrature
using Distributions

import Base: length
import StatsFuns: logsumexp
import SpecialFunctions: loggamma

struct Unscented{R}
    rule::R
end

call_approximate(method::Unscented, q, f) = call_approximate(method, method.rule, q, f)

## Gauss Laguerre ##

struct GaussLaguerre
    length :: Int
end

Base.length(method::GaussLaguerre) = method.length

struct EvaluatedGaussLaguerre{P, W, L}
    points     :: P
    weights    :: W
    logweights :: L
end

function call_approximate(base::Unscented, method::GaussLaguerre, q::Gamma, logf::LogPdf)
    α = shape(q)
    β = rate(q)

    # logg(x) = logf(x/β) - log(x) - log(Γ(α))
    logg = ShiftOutput(Displace(RescaleArguments(logf, inv(β)), Val(-1), log), -loggamma(α))

    points, weights = gausslaguerre(length(method), α)

    evaluated = EvaluatedGaussLaguerre(points, weights, Iterators.map(log, weights))

    return call_approximate(base, evaluated, LogPdf(AutoDomain(), logg))
end

function call_approximate(::Unscented, method::EvaluatedGaussLaguerre, logf::LogPdf)

    points     = method.points
    logweights = method.logweights
    
    # Iterators.map creates a lazy mapping
    logresult = Iterators.map(zip(points, logweights)) do (x, logw)
        # calculate the ln(wi) + logg(xi) terms
        return logw + logf(x)
    end

    # return log sum exp
    return logsumexp(logresult)
end

# TODO for testing pusposes only
function mean_var(base::Unscented, method::GaussLaguerre, q::Gamma, logf::LogPdf)
    α = shape(q)
    β = rate(q)
    iβ = inv(β)
    liβ = log(iβ)
    mlΓα = -loggamma(α)

    points, weights = gausslaguerre(length(method), α)

    # `map` here on purpose as we are going to reuse `logweights``
    evaluated = EvaluatedGaussLaguerre(points, weights, map(log, weights))

    # First we compute a normalisation constant in log-domain
    # logg(x) = logf(x/β) - log(x) - log(Γ(α))
    loggC = (x) -> logf(iβ * x) - log(x) + mlΓα
    logC  = call_approximate(base, evaluated, LogPdf(AutoDomain(), loggC))

    # Second we compute mean using previously computed normalisation constant in log domain
    # loggm(x) = logf(x/β) - log(x) + log(x/b) - log(Γ(α)) - log(C) 
    #          = logf(x/β) + log(1/b) - log(Γ(α)) - log(C)
    loggm = let c = liβ + mlΓα - logC
        (x) -> logf(iβ * x) + c
    end
    logm  = call_approximate(base, evaluated, LogPdf(AutoDomain(), loggm))
    m     = exp(logm)

    # At third we compute var using previously computed C and m in log domain
    # loggv(x) = logf(x/β) - log(x) + 2log(abs(x/b - m)) - log(Γ(α)) - log(C) 
    loggv = let c = mlΓα - logC
        (x) -> begin 
            iβx = iβ * x
            logf(iβx) - log(x) + 2log(abs(iβx - m)) + c
        end
    end
    logv  = call_approximate(base, evaluated, LogPdf(AutoDomain(), loggv))
    v     = exp(logv)
    return m, v
end