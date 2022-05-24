export GaussLaguerre

using FastGaussQuadrature

struct GaussLaguerre{P, W, L}
    points     :: P
    weights    :: W
    logweights :: L
end
