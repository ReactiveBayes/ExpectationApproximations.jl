export Unscented, GaussLaguerre

using FastGaussQuadrature
using Distributions
using LogarithmicNumbers

import Base: length
import LogExpFunctions: logsumexp
import SpecialFunctions: loggamma, gamma

struct Unscented{R}
    rule::R
end

call_approximate(method::Unscented, q, f) = call_approximate(evaluate(method.rule, q), q, f)
call_mean_var(method::Unscented, q, f)    = call_mean_var(evaluate(method.rule, q), q, f)

include("gausslaguerre/gausslaguerre.jl")
