# Approximations without distribution, pure integral
# Supports only `LogFn`
function call_approximate(_, method::GaussLaguerre, ::Nothing, f::LogFn)
    points     = method.points
    logweights = method.logweights
    logf       = log(f)

    # g(x)    = f(x) * e^x
    # logg(x) = logf(x) + x
    logg = (x) -> logf(x) + x

    # Iterators.map creates a lazy mapping
    logresult = Iterators.map(zip(points, logweights)) do (x, logw)
        # calculate the ln(wi) + logg(xi) terms
        return logw + logg(x)
    end

    # return log sum exp in form of the `Logarithmic` number from `LogarithmicNumbers.jl` package
    return exp(Logarithmic, logsumexp(logresult))
end