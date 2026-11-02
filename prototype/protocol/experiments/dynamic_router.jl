# Cost of `DynamicRouter` per call against `FirstOf` (U0.2, item 5 of the plan). Run in the
# universal-dev environment:
#
#     julia --project=~/Projects/Julia/universal-dev dynamic_router.jl

using ProtoBase, ProtoBackends, BenchmarkTools, Distributions, StableRNGs, Printf

const P = ProtoBase
const B = ProtoBackends

function registry(backends...)
    r = P.BackendRegistry()
    foreach(((i, b),) -> P.register_backend!(b; registry = r, priority = i), enumerate(backends))
    return r
end

two = registry(B.ClosedForms(), B.QuantileLegendre(16))
five = registry(B.ClosedForms(), B.GaussHermite(16), B.QuantileLegendre(16), B.QuantileLegendre(8), B.MonteCarlo(100, StableRNG(1)))

q = Gamma(2.0, 1.0)
routers = [
    "FirstOf (default_router)" => B.default_router(16),
    "DynamicRouter, 2 backends" => P.DynamicRouter(two),
    "DynamicRouter, 5 backends" => P.DynamicRouter(five),
    "FirstOf(DynamicRouter), 5 backends" => P.FirstOf(P.DynamicRouter(five)),
]

println("E[log λ] under a Gamma, a closed form: the cost is routing")
for (name, router) in routers
    t = @belapsed P.expectation($router, log, $q)
    a = @ballocated P.expectation($router, log, $q)
    choice = @belapsed P.which_backend($router, typeof(log), typeof($q))
    @printf("  %-36s %8.1f ns per call, %5d bytes; choice alone %8.1f ns\n", name, 1e9 * t, a, 1e9 * choice)
end

println("E[sin λ] under a Gamma, unprepared quadrature with 16 points: routing next to real work")
for (name, router) in routers
    t = @belapsed P.expectation($router, sin, $q)
    @printf("  %-36s %8.1f ns per call\n", name, 1e9 * t)
end
