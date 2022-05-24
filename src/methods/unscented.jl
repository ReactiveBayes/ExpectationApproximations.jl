export Unscented, GaussLaguerre
struct Unscented{R}
    rule :: R
end

call_approximate(method::Unscented, q, f) = call_approximate(method, method.rule, q, f)

using FastGaussQuadrature

struct GaussLaguerre{P, W, L}
    points     :: P
    weights    :: W
    logweights :: L
end

function call_approximate(::Unscented, sigma::GaussLaguerre, q, f::LogPdf)
    error(1)
end