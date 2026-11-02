# %% [markdown]
# # Laplace Approximation: Constrained Variables

# %% [markdown]
# This example is part of the *Universal Inference* section, which explores running inference in a broad class of models by approximating every continuous latent variable with a Gaussian. Many parameters are not free to take any real value: a probability lies between $0$ and $1$, a rate or a precision is positive. A Gaussian on such a variable puts mass where the model is undefined. The standard remedy is to approximate the variable with a Gaussian in an unconstrained space, the logit of a probability or the logarithm of a rate. In the *Gaussian Mixture* example of this section we did this by hand and wrote the model in terms of log-precisions.
#
# Here we do it automatically, without rewriting the model. The model keeps its natural priors, a Beta prior for a probability and a Gamma prior for a rate. A form constraint reads the domain of each variable from the support of its prior, computes the Laplace approximation in the unconstrained coordinates and returns the corresponding distribution on the original domain: a logit-normal for a probability, a log-normal for a rate.
#
# The model is the zero-inflated Poisson regression from the "drunk monks" example of McElreath's *Statistical Rethinking* (2nd edition, Section 12.2). A monastery copies manuscripts at an average rate of $\lambda$ per day. On some days, with probability $p$, the monks drink instead and copy nothing. We observe only the daily number of manuscripts. A zero can come from a day of drinking or from a working day without a finished manuscript, so the two parameters are not separately visible in the data:
#
# $$\begin{aligned}
# p &\sim \mathrm{Beta}(1, 1),\\
# \lambda &\sim \mathrm{Gamma}(2, 1),\\
# y_i &\sim p\,\delta_0 + (1 - p)\,\mathrm{Poisson}(\lambda).
# \end{aligned}$$
#
# No exact message passing rule exists for the likelihood, and none is written in this notebook either.
#
# !!! note
#     The machinery below lives in this notebook on purpose. It is a preview of an upcoming generic approximation layer (ExpectationApproximations.jl and a universal rules package), and will move there once it matures.

# %%
using RxInfer, ReactiveMP, BayesBase, Distributions, LinearAlgebra
using ForwardDiff, FastGaussQuadrature, SpecialFunctions, StatsFuns, StableRNGs, Plots
import MessagePassingRulesBase: NodeFunctionLogPdf, RuleArgs, Target, nodefunction, rule_inputs

# %% [markdown]
# ## The Laplace toolkit
#
# The Laplace approximation of a density $p(x) \propto \exp(\ell(x))$ is the Gaussian
#
# $$\mathcal{N}\left(x \mid \hat{x}, \left(-\nabla^2 \ell(\hat{x})\right)^{-1}\right),\qquad \hat{x} = \arg\max_x \ell(x).$$
#
# The function `laplace` finds the mode $\hat{x}$ with a damped Newton method and returns the Gaussian. Gradients and Hessians come from automatic differentiation (ForwardDiff.jl), so $\ell$ can be any Julia function.

# %%
function laplace(logp, x0::Real; tol = 1.0e-10, maxiters = 100)
    x = float(x0)
    d2(z) = ForwardDiff.derivative(w -> ForwardDiff.derivative(logp, w), z)
    for _ in 1:maxiters
        g, h = ForwardDiff.derivative(logp, x), d2(x)
        h < 0 || error("Laplace: the log-density is not concave at x = $x")
        step = -g / h
        t, fx = 1.0, logp(x)
        while logp(x + t * step) < fx - 1.0e-12 && t > 1.0e-8 # backtracking line search
            t /= 2
        end
        x += t * step
        abs(t * step) < tol && break
    end
    return NormalMeanVariance(x, -inv(d2(x)))
end

function laplace(logp, x0::AbstractVector; tol = 1.0e-10, maxiters = 100)
    x = float.(collect(x0))
    for _ in 1:maxiters
        g, H = ForwardDiff.gradient(logp, x), ForwardDiff.hessian(logp, x)
        F = cholesky(Symmetric(-H); check = false)
        issuccess(F) || error("Laplace: the log-density is not concave at x = $x")
        step = F \ g
        t, fx = 1.0, logp(x)
        while logp(x + t * step) < fx - 1.0e-12 && t > 1.0e-8 # backtracking line search
            t /= 2
        end
        x += t * step
        norm(t * step) < tol && break
    end
    return MvNormalMeanCovariance(x, Matrix(Symmetric(inv(-ForwardDiff.hessian(logp, x)))))
end

# %% [markdown]
# Next, the form constraint. In RxInfer a [form constraint](https://reactivebayes.github.io/ReactiveMP.jl/stable/lib/form-constraints/) is applied to the product of the messages that arrive at a variable. When no closed-form product exists, RxInfer keeps that product as a lazy `ProductOf` object which only knows how to evaluate its `logpdf`; `LaplaceForm` replaces it with its Laplace approximation. Gaussians pass through untouched, so conjugate parts of a model keep their exact updates. The Newton search starts from the mean of the first Gaussian factor in the product, usually the prior, or from zero if the product has no Gaussian factor.

# %%
struct LaplaceForm <: ReactiveMP.AbstractFormConstraint end

ReactiveMP.default_form_check_strategy(::LaplaceForm) = FormConstraintCheckLast()
ReactiveMP.default_prod_constraint(::LaplaceForm) = GenericProd()

ReactiveMP.constrain_form(::LaplaceForm, d::NormalDistributionsFamily) = d
function ReactiveMP.constrain_form(::LaplaceForm, d)
    # A product without a Gaussian factor (a likelihood alone) starts from zero, scalars only
    x0 = something(startpoint(d), 0.0)
    return laplace(x -> logpdf(d, x), x0)
end

startpoint(d::NormalDistributionsFamily) = mean(d)
startpoint(d::ProductOf) = firstpoint((d.left, d.right))
startpoint(d::LinearizedProductOf) = firstpoint(d.vector)
startpoint(d) = nothing

function firstpoint(factors)
    for factor in factors
        x0 = startpoint(factor)
        isnothing(x0) || return x0
    end
    return nothing
end


# %% [markdown]
# ## Laplace in transformed coordinates
#
# Let $x = g(u)$ be a smooth bijection from the real line onto the domain of $x$. If $x$ has the density $p(x)$, then $u = g^{-1}(x)$ has the density
#
# $$p_u(u) = p\left(g(u)\right)\,\left|g'(u)\right|.$$
#
# $p_u$ lives on the whole real line, so its Laplace approximation $\mathcal{N}(\hat{u}, \sigma^2)$ is well defined, and pushing it forward through $g$ gives an approximation of $p(x)$ that respects the domain. We support three domains:
#
# | Domain | $g(u)$ | $\log \lvert g'(u) \rvert$ | Pushforward of $\mathcal{N}(\hat{u}, \sigma^2)$ |
# |---|---|---|---|
# | $(-\infty, \infty)$ | $u$ | $0$ | $\mathcal{N}(\hat{u}, \sigma^2)$ |
# | $(0, \infty)$ | $e^u$ | $u$ | $\mathrm{LogNormal}(\hat{u}, \sigma)$ |
# | $(0, 1)$ | $1 / (1 + e^{-u})$ | $-\log(1 + e^{-u}) - \log(1 + e^{u})$ | $\mathrm{LogitNormal}(\hat{u}, \sigma)$ |
#
# The domain is read from the `support` of the first factor in the product that has one, which is the prior. Gaussian factors give the real line, and for them `TransformedLaplaceForm` coincides with `LaplaceForm`.

# %%
abstract type Domain end
struct RealLine <: Domain end
struct Positive <: Domain end
struct UnitInterval <: Domain end

domain(d::NormalDistributionsFamily) = RealLine()
domain(d::UnivariateDistribution) = (s = support(d); domain(minimum(s), maximum(s)))
domain(d::ProductOf) = firstdomain((d.left, d.right))
domain(d::LinearizedProductOf) = firstdomain(d.vector)
domain(d) = nothing

function domain(lo::Real, hi::Real)
    isinf(lo) && isinf(hi) && return RealLine()
    lo == 0 && isinf(hi) && return Positive()
    lo == 0 && hi == 1 && return UnitInterval()
    error("TransformedLaplaceForm: no transform for the domain ($lo, $hi)")
end

function firstdomain(factors)
    for factor in factors
        D = domain(factor)
        isnothing(D) || return D
    end
    return nothing
end

forward(::RealLine, u) = u
forward(::Positive, u) = exp(u)
forward(::UnitInterval, u) = logistic(u)

inverse(::RealLine, x) = x
inverse(::Positive, x) = log(x)
inverse(::UnitInterval, x) = logit(x)

logjacobian(::RealLine, u) = zero(u)
logjacobian(::Positive, u) = u
logjacobian(::UnitInterval, u) = -log1pexp(-u) - log1pexp(u)

pushforward(::RealLine, g) = g
pushforward(::Positive, g) = LogNormal(mean(g), std(g))
pushforward(::UnitInterval, g) = LogitNormal(mean(g), std(g))

# %% [markdown]
# The Newton search starts from the mean of the prior. We extend the `startpoint` function of the toolkit to any univariate prior, so the untransformed `LaplaceForm` gets the same starting point in the comparison below.

# %%
startpoint(d::ContinuousUnivariateDistribution) = mean(d)

struct TransformedLaplaceForm <: ReactiveMP.AbstractFormConstraint end

ReactiveMP.default_form_check_strategy(::TransformedLaplaceForm) = FormConstraintCheckLast()
ReactiveMP.default_prod_constraint(::TransformedLaplaceForm) = GenericProd()

ReactiveMP.constrain_form(::TransformedLaplaceForm, d::Union{NormalDistributionsFamily, LogNormal, LogitNormal}) = d
function ReactiveMP.constrain_form(::TransformedLaplaceForm, d)
    D = something(domain(d), RealLine())
    x0 = something(startpoint(d), forward(D, 0.0))
    g = laplace(u -> logpdf(d, forward(D, u)) + logjacobian(D, u), inverse(D, x0))
    return pushforward(D, g)
end

# %% [markdown]
# ## A variational rule fallback
#
# The two parameters share every likelihood factor, so we use a mean-field scheme, $q(p, \lambda) = q(p)\,q(\lambda)$. The message from a factor $f$ to $p$ is then $\exp\left(\mathbb{E}_{q(\lambda)}\left[\log f(y_i, p, \lambda)\right]\right)$, and vice versa. As in the *Gaussian Mixture* example, `ExpectedLogPdfFallback` computes these expectations with quadrature where RxInfer's built-in `NodeFunctionRuleFallback` would plug in the mean. The marginals now come in more kinds:
#
# - a log-normal or logit-normal marginal is a Gaussian in transformed coordinates, so Gauss–Hermite quadrature applies there, and the nodes are mapped back;
# - in the first iteration the marginals are the Beta and Gamma priors themselves. For these, and for any other univariate distribution, we use Gauss–Legendre quadrature in probability space: the nodes are quantiles of the distribution.

# %%
using BayesBase, ExpectationApproximations
const EA = ExpectationApproximations

# The notebook's own quadrature: exact sums, Gauss–Hermite for Gaussians and transformed
# Gaussians, Gauss–Legendre in probability space for any other distribution
notebook_router(order) = BayesBase.FirstOf(EA.ExactSum(), EA.GaussHermite(order), EA.QuantileLegendre(order))
import BayesBase: FactorizedJoint

# The fallback of the notebook, with its quadrature replaced by the expectation protocol: one
# preparation per message over the product of the other inputs, any integrand evaluated on it
struct ExpectedLogPdfFallback{B}
    backend::B
end

ExpectedLogPdfFallback(order::Int = 16) = ExpectedLogPdfFallback(notebook_router(order))

(fallback::ExpectedLogPdfFallback)(node, target, args) = nothing

function (fallback::ExpectedLogPdfFallback)(node, ::Target{E}, args::RuleArgs) where {E}
    applicable(nodefunction, node) || return nothing
    inputs = (rule_inputs(node, args.m)..., rule_inputs(node, args.q)...)
    names = map(first, inputs)
    prepared = BayesBase.prepare_expectation(fallback.backend, FactorizedJoint(map(last, inputs)))
    f = nodefunction(node)
    return NodeFunctionLogPdf(x -> BayesBase.expectation(prepared, values -> f(; NamedTuple{names}(values)..., NamedTuple{(E,)}((x,))...)))
end

# %% [markdown]
# Distributions.jl provides no `mean` or `var` for `LogitNormal`, since they have no closed form. We compute moments with the same quadrature.

# %%
const MOMENTS = notebook_router(32)

function moments(q)
    m = BayesBase.expectation(MOMENTS, identity, q)
    return m, sqrt(BayesBase.expectation(MOMENTS, x -> (x - m)^2, q))
end

# %% [markdown]
# ## The likelihood node
#
# The zero-inflated Poisson is a fused node with the interfaces `out`, `p` and `λ`.

# %%
struct ZeroInflatedPoisson{P, L} <: DiscreteUnivariateDistribution
    p::P
    λ::L
end

function BayesBase.logpdf(d::ZeroInflatedPoisson, y)
    y == 0 && return log(d.p + (1 - d.p) * exp(-d.λ))
    return log1p(-d.p) + y * log(d.λ) - d.λ - loggamma(y + 1)
end
BayesBase.insupport(::ZeroInflatedPoisson, y) = true

@define_factor_node(node = ZeroInflatedPoisson, type = Stochastic, interfaces = [:out, :p, :λ])

# %% [markdown]
# ## Data
#
# We follow McElreath's simulation: a year of records, monks who drink on one day in five, and one manuscript per working day on average.

# %%
rng = StableRNG(2024)
N, p_true, λ_true = 365, 0.2, 1.0
drink = rand(rng, N) .< p_true
y = [d ? 0 : rand(rng, Poisson(λ_true)) for d in drink]

println("days with zero manuscripts: ", count(==(0), y), " of ", N, ", of them drinking days: ", count(drink))
histogram(y; bins = -0.5:1:(maximum(y) + 0.5), color = :gray, alpha = 0.6, label = "manuscripts per day", xlabel = "y")

# %% [markdown]
# ## Inference
#
# The model is written exactly as the mathematics above. The only hint about the approximation is the form constraint on each marginal. Mean-field inference also needs a starting point; the priors serve as one.

# %%
@model function monastery(y, prior_p, prior_λ)
    p ~ prior_p
    λ ~ prior_λ
    for i in eachindex(y)
        y[i] ~ ZeroInflatedPoisson(p, λ)
    end
end

prior_p, prior_λ = Beta(1.0, 1.0), Gamma(2.0, 1.0)

transformed_constraints = @constraints begin
    q(p, λ) = q(p)q(λ)
    q(p)::TransformedLaplaceForm()
    q(λ)::TransformedLaplaceForm()
end

initialization = @initialization begin
    q(p) = prior_p
    q(λ) = prior_λ
end

n_iterations = 20

result = infer(
    model = monastery(prior_p = prior_p, prior_λ = prior_λ),
    data = (y = y,),
    constraints = transformed_constraints,
    initialization = initialization,
    iterations = n_iterations,
    returnvars = KeepEach(),
    options = (rulefallback = ExpectedLogPdfFallback(),),
)

q_p, q_λ = result.posteriors[:p][end], result.posteriors[:λ][end]

# %% [markdown]
# The posteriors are a logit-normal for $p$ and a log-normal for $\lambda$, chosen by the supports of the Beta and Gamma priors. The free energy is not available on the fallback path yet, so we monitor convergence through the posterior means.

# %%
mean_p = first.(moments.(result.posteriors[:p]))
mean_λ = first.(moments.(result.posteriors[:λ]))

plot(1:n_iterations, [mean_p mean_λ]; label = ["E[p]" "E[λ]"], xlabel = "iteration", ylabel = "posterior mean", marker = :circle)

# %% [markdown]
# ## References
#
# 1. **The exact posterior.** With two parameters, it can be evaluated on a fine grid.
# 2. **The joint transformed Laplace approximation.** A single Gaussian over $(\operatorname{logit} p, \log \lambda)$, the Laplace approximation of the joint posterior in transformed coordinates, computed directly by Newton's method. It differs from our message passing result only in that it keeps the correlation between $p$ and $\lambda$, which the mean-field factorisation discards.

# %%
logposterior(y, p, λ) = logpdf(prior_p, p) + logpdf(prior_λ, λ) + sum(logpdf(ZeroInflatedPoisson(p, λ), yi) for yi in y)

function grid_posterior(y; ps = range(0.0005, 0.9995; length = 1000), λs = range(0.005, 4; length = 800))
    L = [logposterior(y, p, λ) for p in ps, λ in λs]
    P = exp.(L .- maximum(L))
    P ./= sum(P)
    Pp, Pλ = vec(sum(P; dims = 2)), vec(sum(P; dims = 1))
    mp, mλ = sum(Pp .* ps), sum(Pλ .* λs)
    sp, sλ = sqrt(sum(Pp .* (ps .- mp) .^ 2)), sqrt(sum(Pλ .* (λs .- mλ) .^ 2))
    correlation = sum(P .* ((ps .- mp) * (λs .- mλ)')) / (sp * sλ)
    # marginal densities: probabilities divided by the grid step
    return (ps = ps, λs = λs, density_p = Pp ./ step(ps), density_λ = Pλ ./ step(λs), p = (mp, sp), λ = (mλ, sλ), correlation = correlation)
end

grid = grid_posterior(y)

function joint_laplace(y)
    logdensity(u) = logposterior(y, logistic(u[1]), exp(u[2])) + logjacobian(UnitInterval(), u[1]) + logjacobian(Positive(), u[2])
    g = laplace(logdensity, [logit(mean(prior_p)), log(mean(prior_λ))])
    return LogitNormal(mean(g)[1], sqrt(cov(g)[1, 1])), LogNormal(mean(g)[2], sqrt(cov(g)[2, 2]))
end

joint_p, joint_λ = joint_laplace(y)

# %%
show_moments(label, (m, s)) = string(rpad(label, 22), rpad(round(m; digits = 3), 8), round(s; digits = 3))

println(rpad("", 22), "p: mean  sd      λ: mean  sd")
for (label, a, b) in [
        ("exact (grid)", grid.p, grid.λ),
        ("joint Laplace", moments(joint_p), moments(joint_λ)),
        ("mean-field Laplace", moments(q_p), moments(q_λ)),
    ]
    println(show_moments(label, a), "    ", show_moments("", b)[23:end])
end
println("posterior correlation of p and λ (grid): ", round(grid.correlation; digits = 3))

# %%
plt_p = plot(grid.ps, grid.density_p; color = :black, label = "exact (grid)", xlabel = "p", xlims = (0, 0.5))
plot!(plt_p, grid.ps, pdf.(joint_p, grid.ps); color = :blue, linestyle = :dot, label = "joint Laplace")
plot!(plt_p, grid.ps, pdf.(q_p, grid.ps); color = :red, linestyle = :dash, label = "mean-field Laplace")

plt_λ = plot(grid.λs, grid.density_λ; color = :black, label = "exact (grid)", xlabel = "λ", xlims = (0.6, 1.6))
plot!(plt_λ, grid.λs, pdf.(joint_λ, grid.λs); color = :blue, linestyle = :dot, label = "joint Laplace")
plot!(plt_λ, grid.λs, pdf.(q_λ, grid.λs); color = :red, linestyle = :dash, label = "mean-field Laplace")

plot(plt_p, plt_λ; layout = (1, 2), size = (900, 350), bottom_margin = 5Plots.mm)

# %% [markdown]
# The two approximations make two different errors, and the comparison separates them:
#
# - **Location.** Both Laplace approximations report a larger $p$ than the exact posterior. The posterior of $p$ is skewed, and a Laplace approximation is centred at the mode in transformed coordinates, not at the mean. This is the error of the Laplace approximation itself, and what expectation propagation (moment matching) would correct.
# - **Width.** The joint approximation has nearly the exact standard deviations, the mean-field approximation is about a third too narrow. Zero-inflated models are a hard case for mean field: a zero is explained either by a drinking day or by a working day without a manuscript, so $p$ and $\lambda$ are strongly correlated a posteriori, and a factorised approximation is overconfident. This is the error of the factorisation, and what a joint Gaussian over a block of variables would correct.

# %% [markdown]
# ## Sanity check: a conjugate model
#
# Without zero inflation, the Gamma prior is conjugate to the Poisson likelihood and the posterior of $\lambda$ is $\mathrm{Gamma}(a, b)$ in the shape-rate parametrisation. In log coordinates, $u = \log \lambda$, its log-density is $a u - b e^{u}$ up to a constant, with the mode $\hat{u} = \log(a / b)$ and the curvature $-a$ there. The transformed Laplace approximation is therefore $\mathrm{LogNormal}(\log(a/b), 1/\sqrt{a})$, which the message passing result must reproduce. This model has a single latent variable, so belief propagation and the built-in fallback suffice: every message comes from an observed node and is exact.

# %%
struct PoissonRate{L} <: DiscreteUnivariateDistribution
    λ::L
end

BayesBase.logpdf(d::PoissonRate, y) = y * log(d.λ) - d.λ - loggamma(y + 1)
BayesBase.insupport(::PoissonRate, y) = true

@define_factor_node(node = PoissonRate, type = Stochastic, interfaces = [:out, :λ])

@model function poisson_counts(y, prior_λ)
    λ ~ prior_λ
    for i in eachindex(y)
        y[i] ~ PoissonRate(λ)
    end
end

counts = rand(StableRNG(11), Poisson(λ_true), 20)

conjugate_result = infer(
    model = poisson_counts(prior_λ = prior_λ),
    data = (y = counts,),
    constraints = @constraints(
        begin
            q(λ)::TransformedLaplaceForm()
        end
    ),
    options = (rulefallback = NodeFunctionRuleFallback(),),
)

a, b = shape(prior_λ) + sum(counts), 1 / scale(prior_λ) + length(counts)
exact_λ = Gamma(a, 1 / b)
by_hand = LogNormal(log(a / b), 1 / sqrt(a))

println("message passing: ", conjugate_result.posteriors[:λ])
println("by hand:         ", by_hand)
println()
println(rpad("", 22), "mean    sd")
println(show_moments("exact Gamma", (mean(exact_λ), std(exact_λ))))
println(show_moments("transformed Laplace", (mean(conjugate_result.posteriors[:λ]), std(conjugate_result.posteriors[:λ]))))
println(show_moments("Gaussian Laplace", ((a - 1) / b, sqrt(a - 1) / b))) # mode and curvature of the Gamma in λ

# %% [markdown]
# ## Why the transform matters
#
# What happens with the untransformed `LaplaceForm`, which approximates $p$ and $\lambda$ with Gaussians on the real line, on the same year of data?

# %%
plain_constraints = @constraints begin
    q(p, λ) = q(p)q(λ)
    q(p)::LaplaceForm()
    q(λ)::LaplaceForm()
end

function try_infer(data, constraints)
    try
        result = infer(
            model = monastery(prior_p = prior_p, prior_λ = prior_λ),
            data = (y = data,),
            constraints = constraints,
            initialization = initialization,
            iterations = n_iterations,
            options = (rulefallback = ExpectedLogPdfFallback(),),
        )
        println("inference ran")
        return result
    catch e
        println("inference failed: ", first(split(sprint(showerror, e), '\n'))[1:min(end, 80)], "…")
        return nothing
    end
end

try_infer(y, plain_constraints);

# %% [markdown]
# It fails, for two reasons. The first is that the Newton search for the mode of $q(\lambda)$, started at the prior mean $\lambda = 2$, overshoots below zero, where the product of the messages cannot be evaluated. A more careful line search would fix that. The second reason is fundamental. Under mean field, the message to $\lambda$ is an expectation over $q(p)$, and a Gaussian $q(p)$ has tails below $0$. Quadrature, like any other method of computing that expectation, then evaluates the likelihood at negative probabilities. A Gaussian with the moments of our posterior of $p$ already puts quadrature nodes there:

# %%
gaussian_p = NormalMeanVariance(moments(q_p)[1], moments(q_p)[2]^2)
println("quadrature nodes for a Gaussian q(p) below zero: ", count(<(0), BayesBase.prepare_expectation(EA.GaussHermite(16), gaussian_p).points), " of 16")

# %% [markdown]
# The transformed approximation also behaves well when little data pushes the posterior against a boundary. Take a new monastery with two weeks of records, whose monks rarely drink. The posterior of $p$ then piles up at $p = 0$.

# %%
rng_small = StableRNG(5)
y_small = [rand(rng_small) < 0.05 ? 0 : rand(rng_small, Poisson(λ_true)) for _ in 1:14]
println("data: ", y_small)

grid_small = grid_posterior(y_small; λs = range(0.005, 6; length = 800))
small_result = try_infer(y_small, transformed_constraints)
small_p = small_result.posteriors[:p][end]

println(rpad("", 22), "p: mean  sd")
println(show_moments("exact (grid)", grid_small.p))
println(show_moments("mean-field Laplace", moments(small_p)))

# %%
plt_small = plot(grid_small.ps, grid_small.density_p; color = :black, label = "exact (grid)", xlabel = "p", ylabel = "density")
plot!(plt_small, grid_small.ps, pdf.(small_p, grid_small.ps); color = :red, linestyle = :dash, label = "mean-field Laplace")
plt_small

# %% [markdown]
# Inference runs, and the result respects the domain, but it is visibly a Laplace approximation. The exact density has its maximum at $p = 0$, while every logit-normal density vanishes there. The transform makes the approximation well defined, not exact; at a boundary, a family that can put its mode there, such as a Beta distribution, would fit better.

# %% [markdown]
# ## Exact vs approximate, and what did not work
#
# - **What worked.** The model was written with its natural Beta and Gamma priors and was not rewritten in unconstrained coordinates. The form constraint chose the logit and log transforms from the supports of the priors and returned logit-normal and log-normal posteriors, which converge in a few iterations. On the conjugate Poisson-Gamma model the result equals the Laplace approximation computed by hand in log coordinates, and it is closer to the exact posterior than the Gaussian Laplace approximation, because the log-normal is skewed in the right direction.
# - **Why the transform.** Without it, inference does not run at all, even with a year of data: a Gaussian marginal of a probability has tails below zero, and the variational messages evaluate the likelihood there. With it, inference runs also when two weeks of data push the posterior against the boundary.
# - **Accuracy.** The approximation is not exact, and the comparison with the grid and with the joint Laplace approximation shows where the error comes from. The means are those of the mode in transformed coordinates; the posterior is skewed and the means are off, which moment matching would correct. The standard deviations of the mean-field approximation are too small, because the posterior correlation of $p$ and $\lambda$ is strong; the joint approximation, a single Gaussian over $(\operatorname{logit} p, \log \lambda)$, gets them nearly right. Message passing over such a block needs a transform per component of a vector variable, which our form constraint does not support yet.
# - **What we had to work around.**
#     - The marginals of the first iteration are the priors, Beta and Gamma, so the fallback needs a quadrature for any distribution, not only for Gaussians.
#     - Mean-field inference does not start without an initialisation; we used the priors.
#     - Distributions.jl has no `mean` or `var` for `LogitNormal`, and BayesBase has no `mean(log, ::LogNormal)`, although it is simply the location parameter. The exact rules of RxInfer that consume a precision need only its mean and accept a log-normal marginal; a rule that needs $\mathbb{E}[\log \lambda]$ would fail.
#     - The domain comes from the support of a prior. A constrained variable without such a factor, for example the output of a deterministic function, has no domain to read.
#     - As in the other examples of this section, the free energy is not available on the fallback path.
