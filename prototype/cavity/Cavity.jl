# A small factor-graph engine for design B of the cavity methods (design/cavity_propagation.md):
# every variable keeps one Gaussian site per edge, proj(m_k · c_k) / c_k against the cavity c_k,
# its outbound message on that edge, and every belief is a product of sites. The engine knows no
# model; a site is computed by one function, `compute_site`, which is the stage ReactiveMP would
# call. Run in the universal-dev environment.

module Cavity

using BayesBase, Distributions, LinearAlgebra
using StatsFuns: log2π
import ExpectationApproximations as EA

export NP, Graph, Prior, LinearGaussian, Observation, Delta,
    LaplacePropagation, ExpectationPropagation, CVIPropagation,
    variable!, factor!, run!, marginal, free_energy, local_free_energy, isproper

## Gaussians in natural parameters: exp(hᵀx - xᵀJx/2), with no constant, J possibly indefinite

struct NP
    h::Vector{Float64}
    J::Matrix{Float64}
end

NP(d::Integer) = NP(zeros(d), zeros(d, d))
NP(q::AbstractMvNormal) = (J = Matrix(inv(Symmetric(cov(q)))); NP(J * mean(q), J))
# from a mean and a covariance; `NP(h, J)` itself takes natural parameters
frommoments(m::AbstractVector, V::AbstractMatrix) = (J = Matrix(inv(Symmetric(V))); NP(J * m, J))

Base.:+(a::NP, b::NP) = NP(a.h + b.h, a.J + b.J)
Base.:-(a::NP, b::NP) = NP(a.h - b.h, a.J - b.J)
Base.length(a::NP) = length(a.h)
mix(β, previous::NP, new::NP) = NP(β * previous.h + (1 - β) * new.h, β * previous.J + (1 - β) * new.J)
distance(a::NP, b::NP) = max(maximum(abs, a.h - b.h; init = 0.0), maximum(abs, a.J - b.J; init = 0.0))

isfinite_np(a::NP) = all(isfinite, a.h) && all(isfinite, a.J)
isproper(a::NP) = isfinite_np(a) && isposdef(Symmetric(a.J))
isindefinite(a::NP) = !isfinite_np(a) || minimum(eigvals(Symmetric(a.J))) < -1.0e-10 * max(1.0, maximum(abs, a.J))
logkernel(a::NP, x) = dot(a.h, x) - dot(x, a.J * x) / 2
# log ∫ exp(hᵀx - xᵀJx/2) dx, for a proper one
lognormaliser(a::NP) = (V = inv(Symmetric(a.J)); (length(a) * log2π + logdet(V) + dot(a.h, V * a.h)) / 2)
function distribution(a::NP)
    V = Matrix(inv(Symmetric(a.J)))
    V = (V + V') / 2
    return MvNormal(V * a.h, V)
end

const DEBUG = Ref(false)

## Messages from a factor: a Gaussian (closed), a log-density (raw), or a pushforward (raw, O1)

struct Raw{F}
    logf::F
end

struct Pushforward{D, G}
    cx::D
    g::G
end

## Factors: `message(factor, slot, cavities)` is BP's message on edge `slot`, from the cavities on
## every edge, the one of `slot` unused; `logZ_closed` is log ∫ f ∏ c for an all-Gaussian factor

abstract type Factor end

"A Gaussian factor of one variable, `N(x | m, V)`, or a Gaussian likelihood written as one."
struct Prior <: Factor
    q::MvNormal
end
message(f::Prior, slot, cavities) = NP(f.q)
function logZ_closed(f::Prior, cavities)
    factor = NP(f.q)
    k = -(length(f.q) * log2π + logdet(cov(f.q)) + dot(mean(f.q), factor.h)) / 2
    return k + lognormaliser(factor + cavities[1])
end

"`out ~ N(A in + b, Q)`: slot 1 is `out`, slot 2 is `in`, closed both ways."
struct LinearGaussian <: Factor
    A::Matrix{Float64}
    b::Vector{Float64}
    Q::Matrix{Float64}
end
# The factor's joint natural parameters over (out, in), and its log-constant
function joint(f::LinearGaussian)
    W = inv(f.Q)
    J = [W -W*f.A; -f.A'*W f.A'*W*f.A]
    h = [W * f.b; -f.A' * W * f.b]
    k = -(size(f.Q, 1) * log2π + logdet(f.Q) + dot(f.b, W * f.b)) / 2
    return h, J, k
end
function message(f::LinearGaussian, slot, cavities)
    h, J, _ = joint(f)
    dout = size(f.A, 1)
    a, b = slot == 1 ? (1:dout, (dout + 1):length(h)) : ((dout + 1):length(h), 1:dout)
    c = cavities[3 - slot]
    S = J[b, b] + c.J
    return NP(h[a] - J[a, b] * (S \ (h[b] + c.h)), J[a, a] - J[a, b] * (S \ J[b, a]))
end
function logZ_closed(f::LinearGaussian, cavities)
    h, J, k = joint(f)
    return k + lognormaliser(NP(h, J) + NP([cavities[1].h; cavities[2].h], cat(cavities[1].J, cavities[2].J; dims = (1, 2))))
end

"A factor `exp(logf(x))` of one variable, with no rules: always raw."
struct Observation{F} <: Factor
    logf::F
end
message(f::Observation, slot, cavities) = Raw(f.logf)

"""
`out = g(in)`: slot 1 is `out`, slot 2 is `in`. Backwards it is raw, the cavity of `out` pulled
back; forwards it is the moment-matched pushforward of the cavity of `in`, as the fallback sends
it today, or, with `forward_site`, a raw pushforward whose site is computed against `out`'s cavity
(open question O1).
"""
struct Delta{G} <: Factor
    g::G
    forward_site::Bool
    n::Int
end
Delta(g; forward_site = false, n = 20) = Delta(g, forward_site, n)
vec_output(g, x) = (y = g(x); y isa Number ? [y] : collect(y))
function message(f::Delta, slot, cavities)
    if slot == 2
        cz = cavities[1]
        return Raw(x -> logkernel(cz, vec_output(f.g, x)))
    end
    cx = cavities[2]
    isproper(cx) || return nothing
    f.forward_site && return Pushforward(distribution(cx), f.g)
    m, V = EA.mean_cov(EA.GaussHermite(f.n), x -> vec_output(f.g, x), distribution(cx))
    if !isposdef(Symmetric(V))
        DEBUG[] && @warn "degenerate pushforward" cx m V
        return nothing
    end
    return frommoments(m, V)
end

## Site policies: the projection of the tilted density `exp(logf) · cavity`, with its log-normaliser
## against the normalised cavity

struct LaplacePropagation end
"""
Moment matching by Gauss–Hermite with `n` points per axis. `center = :cavity` puts the points on
the cavity, as EA's `tilted_mean_cov` does; `center = :laplace` puts them on the tilted density's
Laplace approximation and weighs them by the tilted density against it, which resolves a factor
much sharper than its cavity.
"""
struct ExpectationPropagation
    n::Int
    center::Symbol
end
ExpectationPropagation(n = 20; center = :laplace) = ExpectationPropagation(n, center)
struct CVIPropagation end

function project_logf(::LaplacePropagation, logf, cavity::MvNormal)
    r = EA.laplace(x -> logf(x) + logpdf(cavity, x), mean(cavity))
    return (; q = r.distribution, r.logZ, r.converged)
end
function project_logf(p::ExpectationPropagation, logf, cavity::MvNormal)
    if p.center === :cavity
        logZ, m, V = EA.tilted_mean_cov(EA.GaussHermite(p.n), logf, cavity)
        return (; q = MvNormal(collect(m), Symmetric((V + V') / 2)), logZ, converged = isfinite(logZ))
    end
    logtilted(x) = logf(x) + logpdf(cavity, x)
    laplace = EA.laplace(logtilted, mean(cavity))
    base = laplace.converged ? laplace.distribution : cavity
    set = BayesBase.prepare_expectation(EA.GaussHermite(p.n), base)
    xs = [collect(x) for x in set.points]
    logw = [log(w) + logtilted(x) - logpdf(base, x) for (x, w) in zip(xs, set.weights)]
    shift = maximum(logw)
    w = exp.(logw .- shift)
    total = sum(w)
    w ./= total
    m = sum(w .* xs)
    V = sum(wi * (x - m) * (x - m)' for (wi, x) in zip(w, xs))
    return (; q = MvNormal(m, Symmetric((V + V') / 2)), logZ = shift + log(total), converged = isfinite(total))
end
function project_logf(::CVIPropagation, logf, cavity::MvNormal)
    r = EA.gaussian_variational(x -> logf(x) + logpdf(cavity, x), mean(cavity))
    return (; q = r.distribution, r.logZ, r.converged)
end

# O1: the site of a pushforward of `cx` through `g`, against `out`'s cavity, by the moments of
# g(x) under cx(x) · cavity(g(x)), whatever the policy; Laplace's method has no density to use.
# The pushforward is proper, so the cavity need not be: it weighs the points as a kernel.
function project(policy, message::Pushforward, cavity::NP)
    n = policy isa ExpectationPropagation ? policy.n : 20
    set = BayesBase.prepare_expectation(EA.GaussHermite(n), message.cx)
    zs = [vec_output(message.g, x) for x in set.points]
    logw = [log(w) + logkernel(cavity, z) for (z, w) in zip(zs, set.weights)]
    shift = maximum(logw)
    w = exp.(logw .- shift)
    total = sum(w)
    w ./= total
    m = sum(w .* zs)
    V = sum(wi * (z - m) * (z - m)' for (wi, z) in zip(w, zs))
    return (; q = MvNormal(m, Symmetric((V + V') / 2)), logZ = shift + log(total), converged = true)
end
project(policy, message::Raw, cavity::MvNormal) = project_logf(policy, message.logf, cavity)

"""
    compute_site(policy, message, cavity, previous) -> (site, outcome)

The stage design B adds to a variable: a closed message is its own site; a raw one gets
`proj(message · cavity) / cavity`, or keeps `previous` while its cavity is improper or when the
projection fails. `outcome` is `:closed`, `:projected`, `:waiting` or `:failed`, with `logZ`, the
log-normaliser of the tilted density against the normalised cavity.
"""
compute_site(policy, message::NP, cavity::NP, previous) = (message, (; status = :closed, logZ = NaN))
# what a projection takes as its cavity: a distribution, which must be proper, or for a
# pushforward the natural parameters themselves
cavity_for(message, cavity::NP) = isproper(cavity) ? distribution(cavity) : nothing
cavity_for(::Pushforward, cavity::NP) = isfinite_np(cavity) ? cavity : nothing

function compute_site(policy, message, cavity::NP, previous)
    c = cavity_for(message, cavity)
    isnothing(c) && return (previous, (; status = :waiting, logZ = NaN))
    result = try
        project(policy, message, c)
    catch error
        error isa InterruptException && rethrow()
        nothing
    end
    if isnothing(result) || !result.converged || !isfinite(result.logZ) || !all(isfinite, cov(result.q))
        return (previous, (; status = :failed, logZ = NaN))
    end
    site = NP(result.q) - cavity
    isfinite_np(site) || return (previous, (; status = :failed, logZ = NaN))
    return (site, (; status = :projected, result.logZ))
end

## The graph

mutable struct Edge
    variable::Int
    factor::Int
    slot::Int
    inbound::Any        # the factor's latest message
    site::NP
    cavity::NP          # the variable's outbound message on this edge
    logZ::Float64       # of the latest projected site
    status::Symbol
end

struct Variable
    name::Any
    dim::Int
    edges::Vector{Int}
end

struct FactorNode
    factor::Factor
    edges::Vector{Int}   # by slot
end

Base.@kwdef mutable struct Counters
    projections::Int = 0
    failed::Int = 0
    waiting::Int = 0
    improper_cavities::Int = 0   # O3: outbound messages that were indefinite when sent
    improper_receivers::Dict{Symbol, Int} = Dict{Symbol, Int}()  # O3: the factors that got them
    improper_marginals::Int = 0  # O2
    clipped::Int = 0
end

struct Graph
    variables::Vector{Variable}
    factors::Vector{FactorNode}
    edges::Vector{Edge}
    names::Dict{Any, Int}
    counters::Counters
end
Graph() = Graph(Variable[], FactorNode[], Edge[], Dict{Any, Int}(), Counters())

function variable!(graph::Graph, name, dim::Integer)
    push!(graph.variables, Variable(name, dim, Int[]))
    graph.names[name] = length(graph.variables)
    return name
end

function factor!(graph::Graph, factor::Factor, names...)
    fi = length(graph.factors) + 1
    edges = Int[]
    for (slot, name) in enumerate(names)
        vi = graph.names[name]
        d = graph.variables[vi].dim
        push!(graph.edges, Edge(vi, fi, slot, nothing, NP(d), NP(d), NaN, :none))
        push!(edges, length(graph.edges))
        push!(graph.variables[vi].edges, length(graph.edges))
    end
    push!(graph.factors, FactorNode(factor, edges))
    return fi
end

belief(graph::Graph, vi::Int) = reduce(+, (graph.edges[e].site for e in graph.variables[vi].edges))
marginal(graph::Graph, name) = distribution(belief(graph, graph.names[name]))

## One step of each kind

function update_inbound!(graph::Graph, fi::Int)
    node = graph.factors[fi]
    cavities = [graph.edges[e].cavity for e in node.edges]
    for (slot, e) in enumerate(node.edges)
        m = message(node.factor, slot, cavities)
        isnothing(m) || (graph.edges[e].inbound = m)
    end
    return nothing
end

# Sites one raw edge after another, each against the cavity of the current sites, then every
# outbound message. `improper = :skip` keeps a site that would make the marginal improper;
# `:clip` floors the marginal's eigenvalues and gives the difference to that site.
function update_variable!(graph::Graph, vi::Int, policy; damping = 0.0, improper = :skip, floor = 1.0e-8)
    v, c = graph.variables[vi], graph.counters
    for e in v.edges
        edge = graph.edges[e]
        isnothing(edge.inbound) && continue
        cavity = reduce(+, (graph.edges[j].site for j in v.edges if j != e); init = NP(v.dim))
        site, outcome = compute_site(policy, edge.inbound, cavity, edge.site)
        if outcome.status === :projected
            c.projections += 1
            site = damping > 0 && edge.status === :projected ? mix(damping, edge.site, site) : site
            edge.logZ = outcome.logZ
        end
        outcome.status === :failed && (c.failed += 1)
        outcome.status === :waiting && (c.waiting += 1)
        edge.site = site
        # a site that waits or fails keeps the status of the one it keeps
        edge.status = outcome.status in (:closed, :projected) || edge.status === :none ? outcome.status : edge.status
    end
    q = belief(graph, vi)
    if !isproper(q)
        c.improper_marginals += 1
        if improper === :clip
            F = eigen(Symmetric(q.J))
            Jq = F.vectors * Diagonal(max.(F.values, floor)) * F.vectors'
            raw = findlast(e -> graph.edges[e].status === :projected, v.edges)
            if !isnothing(raw)
                e = v.edges[raw]
                graph.edges[e].site = NP(graph.edges[e].site.h, graph.edges[e].site.J + (Jq - q.J))
                c.clipped += 1
            end
        end
    end
    for e in v.edges
        cavity = reduce(+, (graph.edges[j].site for j in v.edges if j != e); init = NP(v.dim))
        # O3: an indefinite outbound message to a factor that passes it on; a singular one, which
        # says nothing about some direction, is an ordinary message
        receiver = graph.factors[graph.edges[e].factor].factor
        if isindefinite(cavity) && length(graph.factors[graph.edges[e].factor].edges) > 1
            c.improper_cavities += 1
            kind = nameof(typeof(receiver))
            c.improper_receivers[kind] = get(c.improper_receivers, kind, 0) + 1
        end
        graph.edges[e].cavity = cavity
    end
    return nothing
end

"""
    run!(graph, policy; schedule = :parallel, order = nothing, iterations = 100, tol = 1e-9,
         damping = 0.0, improper = :skip) -> (; iterations, converged, history)

Iterate to a fixed point of the marginals. `:parallel` computes every factor's messages from the
current cavities, then every variable's sites; `:sweep` visits the variables of `order` forwards
and then backwards, each after the messages of its factors.
"""
function run!(graph::Graph, policy; schedule = :parallel, order = nothing, iterations = 100, tol = 1.0e-9, damping = 0.0, improper = :skip)
    order = isnothing(order) ? collect(1:length(graph.variables)) : [graph.names[n] for n in order]
    previous = [belief(graph, vi) for vi in eachindex(graph.variables)]
    history = Float64[]
    for iteration in 1:iterations
        if schedule === :parallel
            foreach(fi -> update_inbound!(graph, fi), eachindex(graph.factors))
            foreach(vi -> update_variable!(graph, vi, policy; damping, improper), eachindex(graph.variables))
        else
            for vi in [order; reverse(order)]
                foreach(e -> update_inbound!(graph, graph.edges[e].factor), graph.variables[vi].edges)
                update_variable!(graph, vi, policy; damping, improper)
            end
        end
        current = [belief(graph, vi) for vi in eachindex(graph.variables)]
        δ = maximum(distance.(current, previous))
        push!(history, δ)
        previous = current
        if δ < tol && all(e -> !isnothing(e.inbound) && e.status !== :waiting, graph.edges)
            return (; iterations = iteration, converged = true, history)
        end
    end
    return (; iterations, converged = false, history)
end

"""
    free_energy(graph) -> Float64

The EP free energy in Bethe form, `-(Σ_a log Z_a - Σ_i (d_i - 1) log ∫ q_i)`, with the sites in
place of BP's messages and `q_i` the product of variable i's sites. A factor with a raw edge
takes its `log Z_a` from that edge's projection, `log ∫ exp(logf) c + log ∫ c`; an all-Gaussian
factor computes it in closed form. Exact for a tree of Gaussians.
"""
function free_energy(graph::Graph)
    logZ = 0.0
    for node in graph.factors
        # a log-density's edge, whose log Z is against the normalised cavity
        raw = findfirst(e -> graph.edges[e].status === :projected && graph.edges[e].inbound isa Raw, node.edges)
        if isnothing(raw)
            logZ += logZ_closed(node.factor, [graph.edges[e].cavity for e in node.edges])
        else
            edge = graph.edges[node.edges[raw]]
            logZ += edge.logZ + lognormaliser(edge.cavity)
        end
    end
    for (vi, v) in enumerate(graph.variables)
        logZ -= (length(v.edges) - 1) * lognormaliser(belief(graph, vi))
    end
    return -logZ
end

"""
    local_free_energy(graph, policy) -> Float64

The same free energy as [`free_energy`](@ref), in terms that each node and each variable computes
from what it sees, as ReactiveMP would. Bethe's terms for the variables, `(d_i - 1) H[q_i]`, and for
the nodes with rules, `E_b[-log f_a] - H[b_a]` with `b_a ∝ f_a ∏ c_ia`, stay as they are. Only a
node that sends a raw message takes a new term,

    F_a = -log ∫ f_a ∏_i c_ia + Σ_i E_{q_i}[log c_ia],

from its inbound messages `c_ia` (the cavities) and the marginals `q_i`. It does not change when a
`c_ia` is rescaled, so the cavities enter as kernels, improper ones included. The integral is the
policy's own estimate: Laplace's for Laplace propagation, the moments' quadrature for EP.
"""
function local_free_energy(graph::Graph, policy)
    F = 0.0
    for node in graph.factors
        cavities = [graph.edges[e].cavity for e in node.edges]
        if node.factor isa Union{Prior, LinearGaussian}
            F += bethe_term(node.factor, cavities)
        else
            raw = findfirst(e -> graph.edges[e].inbound isa Raw, node.edges)
            edge = graph.edges[node.edges[raw]]
            # log ∫ f ∏ c with the cavities as kernels: the raw message carries the other cavities
            # as kernels, and its own edge's enters here, so no cavity needs to be proper
            start = mean(distribution(belief(graph, edge.variable)))
            F -= log_integral(policy, x -> edge.inbound.logf(x) + logkernel(edge.cavity, x), start)
            for (e, c) in zip(node.edges, cavities)
                F += expected_logkernel(distribution(belief(graph, graph.edges[e].variable)), c)
            end
        end
    end
    for (vi, v) in enumerate(graph.variables)
        F += (length(v.edges) - 1) * entropy(distribution(belief(graph, vi)))
    end
    return F
end

# log ∫ exp(logp), as each policy estimates it: Laplace's method, the variational bound, or
# Gauss–Hermite points on the Laplace approximation, weighted against it
log_integral(::LaplacePropagation, logp, start) = EA.laplace(logp, start).logZ
log_integral(::CVIPropagation, logp, start) = EA.gaussian_variational(logp, start).logZ
log_integral(p::ExpectationPropagation, logp, start) = log_integral(logp, start; n = p.n)
function log_integral(logp, start; n = 30)
    base = EA.laplace(logp, start).distribution
    set = BayesBase.prepare_expectation(EA.GaussHermite(n), base)
    logw = [log(w) + logp(collect(x)) - logpdf(base, collect(x)) for (x, w) in zip(set.points, set.weights)]
    top = maximum(logw)
    return top + log(sum(exp, logw .- top))
end

expected_logkernel(q, c::NP) = dot(c.h, mean(q)) - (tr(c.J * cov(q)) + dot(mean(q), c.J * mean(q))) / 2

# E_b[-log f] - H[b], with b ∝ f ∏ c, the node's belief
function bethe_term(f::Prior, cavities)
    b = distribution(NP(f.q) + cavities[1])
    return -expected_logpdf(f.q, b) - entropy(b)
end
function bethe_term(f::LinearGaussian, cavities)
    h, J, k = joint(f)
    b = distribution(NP(h, J) + NP(reduce(vcat, (c.h for c in cavities)), cat((c.J for c in cavities)...; dims = (1, 2))))
    logf_expectation = k + dot(h, mean(b)) - (tr(J * cov(b)) + dot(mean(b), J * mean(b))) / 2
    return -logf_expectation - entropy(b)
end
expected_logpdf(p::MvNormal, q) = -(length(p) * log2π + logdet(cov(p)) + tr(cov(p) \ cov(q)) + dot(mean(q) - mean(p), cov(p) \ (mean(q) - mean(p)))) / 2

end
