# %% [markdown]
# # Laplace Approximation: Gaussian Mixture

# %% [markdown]
# This example is part of the *Universal Inference* section, which explores running inference in a broad class of models by approximating every continuous latent variable with a Gaussian. Real models rarely consist of unconstrained continuous variables only. A Gaussian mixture has two other kinds:
#
# - **discrete variables**, the component assignments $z_i$, which have a finite number of values. They need no approximation: their expectations are exact sums.
# - **constrained variables**, the component precisions, which must be positive. We make them Gaussian in an *unconstrained* space by modelling the log-precisions $s_k$, so that the precision is $e^{s_k} > 0$ for every value of $s_k$.
#
# The model is
#
# $$\begin{aligned}
# \pi &\sim \mathrm{Dirichlet}(1, \ldots, 1),\\
# m &\sim \mathcal{N}(0, 100 I),\qquad s \sim \mathcal{N}(0, 4 I),\\
# z_i &\sim \mathrm{Categorical}(\pi),\\
# y_i &\sim \mathcal{N}\left(m_{z_i}, e^{-s_{z_i}}\right).
# \end{aligned}$$
#
# The Dirichlet-Categorical pair is conjugate, so RxInfer keeps its exact rules for it. Everything else is handled by generic machinery written in this notebook: the Laplace toolkit from the first example of this section, a projection for discrete variables, and a rule fallback that computes proper variational messages. We compare the result with RxInfer's hand-derived conjugate solution for the same kind of model.
#
# !!! note
#     The machinery below lives in this notebook on purpose. It is a preview of an upcoming generic approximation layer (ExpectationApproximations.jl and a universal rules package), and will move there once it matures.

# %%
using RxInfer, ReactiveMP, BayesBase, Distributions, LinearAlgebra
using ForwardDiff, FastGaussQuadrature, StatsFuns, StableRNGs, Plots
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
# ## Discrete variables: an exact projection
#
# The marginal of $z_i$ is the product of a categorical message (from $\pi$) and a `logpdf` message from the likelihood. Since $z_i$ takes only $K$ values, the product can be normalised exactly by evaluating it at each value. `CategoricalForm` does that.

# %%
struct CategoricalForm <: ReactiveMP.AbstractFormConstraint end

ReactiveMP.default_form_check_strategy(::CategoricalForm) = FormConstraintCheckLast()
ReactiveMP.default_prod_constraint(::CategoricalForm) = GenericProd()

ReactiveMP.constrain_form(::CategoricalForm, d::Categorical) = d
function ReactiveMP.constrain_form(::CategoricalForm, d)
    K = ncategories_in(d)
    isnothing(K) && error("CategoricalForm: no categorical factor in the product")
    return Categorical(softmax([logpdf(d, k) for k in 1:K]))
end

ncategories_in(d::Categorical) = ncategories(d)
ncategories_in(d::ProductOf) = (K = ncategories_in(d.left); isnothing(K) ? ncategories_in(d.right) : K)
ncategories_in(d) = nothing

# %% [markdown]
# ## A variational rule fallback
#
# In a mean-field (variational) scheme, the message from a factor $f$ to a variable $x$ is $\exp\left(\mathbb{E}_{q}\left[\log f(x, \ldots)\right]\right)$, where the expectation is over the marginals of all other variables of the factor. RxInfer's built-in `NodeFunctionRuleFallback` replaces the expectation with a plug-in: it evaluates $\log f$ at the *means* of the other marginals. That ignores their uncertainty, and it breaks for discrete variables, because the mean of a categorical distribution over $\{1, \ldots, K\}$ is a number between $1$ and $K$, not a component index.
#
# `ExpectedLogPdfFallback` computes the expectation instead:
#
# - over a categorical marginal, exactly, as a weighted sum over its $K$ values;
# - over a Gaussian marginal, with Gauss–Hermite quadrature (a product rule in several dimensions, fine for a handful of dimensions);
# - over an observed value, by plugging it in.
#
# Like the built-in fallback, it is only consulted where no exact rule exists.

# %%
using BayesBase, ProtoBackends
import BayesBase: FactorizedJoint

# The fallback of the notebook, with its quadrature replaced by the expectation protocol: one
# preparation per message over the product of the other inputs, any integrand evaluated on it
struct ExpectedLogPdfFallback{B}
    backend::B
end

ExpectedLogPdfFallback(order::Int = 5) = ExpectedLogPdfFallback(ProtoBackends.default_router(order))

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
# ## The likelihood node
#
# The mixture likelihood is a fused node with the interfaces `out`, `z`, `m` and `s`. Its `logpdf` accepts the assignment either as a component index or as a vector of weights.

# %%
struct LogPrecisionMixture{Z, M, S} <: ContinuousUnivariateDistribution
    z::Z
    m::M
    s::S
end

component_weights(z::Integer, K) = (1:K) .== z
component_weights(z::AbstractVector, K) = z

function BayesBase.logpdf(d::LogPrecisionMixture, y)
    w = component_weights(d.z, length(d.m))
    return sum(w[k] * (d.s[k] - log2π - exp(d.s[k]) * (y - d.m[k])^2) / 2 for k in eachindex(d.m))
end
BayesBase.insupport(::LogPrecisionMixture, y) = true

@define_factor_node(node = LogPrecisionMixture, type = Stochastic, interfaces = [:out, :z, :m, :s])

# %% [markdown]
# ## Data

# %%
rng = StableRNG(42)
N, K = 300, 2
true_weights, true_means, true_sds = [0.4, 0.6], [-2.0, 2.0], [0.5, 1.0]
z_true = rand(rng, Categorical(true_weights), N)
y = [rand(rng, Normal(true_means[z], true_sds[z])) for z in z_true]

histogram(y; bins = 40, normalize = :pdf, color = :gray, alpha = 0.5, label = "data", xlabel = "y")

# %% [markdown]
# ## Inference
#
# The model is written with vectors $m$ and $s$ of length $K$, so the same code works for any number of components.

# %%
@model function laplace_mixture(y, K)
    π ~ Dirichlet(ones(K))
    m ~ MvNormalMeanCovariance(zeros(K), 100.0 * diageye(K))
    s ~ MvNormalMeanCovariance(zeros(K), 4.0 * diageye(K))
    for i in eachindex(y)
        z[i] ~ Categorical(π)
        y[i] ~ LogPrecisionMixture(z[i], m, s)
    end
end

laplace_constraints = @constraints begin
    q(π, m, s, z) = q(π)q(m)q(s)q(z)
    q(m)::LaplaceForm()
    q(s)::LaplaceForm()
    q(z)::CategoricalForm()
end

# The initial means differ to break the symmetry between the components
laplace_initialization = @initialization begin
    q(π) = Dirichlet(ones(K))
    q(m) = MvNormalMeanCovariance([-1.0, 1.0], diageye(K))
    q(s) = MvNormalMeanCovariance(zeros(K), diageye(K))
    q(z) = Categorical(fill(1 / K, K))
end

n_iterations = 30

result = infer(
    model = laplace_mixture(K = K),
    data = (y = y,),
    constraints = laplace_constraints,
    initialization = laplace_initialization,
    iterations = n_iterations,
    returnvars = (m = KeepEach(), s = KeepEach(), π = KeepLast()),
    options = (rulefallback = ExpectedLogPdfFallback(),),
)

q_m, q_s, q_π = result.posteriors[:m][end], result.posteriors[:s][end], result.posteriors[:π]

# %% [markdown]
# The free energy is not available on the fallback path yet (see the end of this notebook), so we monitor convergence through the posterior means of the components.

# %%
plot(1:n_iterations, reduce(hcat, mean.(result.posteriors[:m]))'; label = ["E[m₁]" "E[m₂]"], xlabel = "iteration", ylabel = "posterior mean", marker = :circle)

# %% [markdown]
# ## Reference: the conjugate model
#
# RxInfer has exact variational rules for a Gaussian mixture with Gamma-distributed precisions, the `NormalMixture` node. Its priors differ slightly from ours (a Gamma prior on the precision instead of a Gaussian prior on the log-precision), so we expect close but not identical results.

# %%
@model function conjugate_mixture(y, K)
    local m
    local w
    π ~ Dirichlet(ones(K))
    for k in 1:K
        m[k] ~ NormalMeanVariance(0.0, 100.0)
        w[k] ~ GammaShapeRate(1.0, 1.0)
    end
    for i in eachindex(y)
        z[i] ~ Categorical(π)
        y[i] ~ NormalMixture(switch = z[i], m = m, p = w)
    end
end

conjugate_initialization = @initialization begin
    q(π) = Dirichlet(ones(K))
    q(m) = [NormalMeanVariance(-1.0, 1.0), NormalMeanVariance(1.0, 1.0)]
    q(w) = [GammaShapeRate(1.0, 1.0), GammaShapeRate(1.0, 1.0)]
end

result_conjugate = infer(
    model = conjugate_mixture(K = K),
    data = (y = y,),
    constraints = MeanField(),
    initialization = conjugate_initialization,
    iterations = n_iterations,
    returnvars = KeepLast(),
)

# %% [markdown]
# We report the component standard deviation at the posterior mean of the log-precision, $e^{-\mathbb{E}[s_k]/2}$, and for the conjugate model at the posterior mean of the precision, $\mathbb{E}[w_k]^{-1/2}$.

# %%
row(name, means, sds, component_sds, weights) = println(
    rpad(name, 22),
    "m = ", round.(means; digits = 3), "  sd(m) = ", round.(sds; digits = 3),
    "  component sd = ", round.(component_sds; digits = 3), "  π = ", round.(weights; digits = 3)
)

row("truth", true_means, [0.0, 0.0], true_sds, true_weights)
row("empirical (true z)", [sum(y[z_true .== k]) / count(z_true .== k) for k in 1:K], [0.0, 0.0], [std(y[z_true .== k]) for k in 1:K], [count(z_true .== k) / N for k in 1:K])
row("Laplace + exact z", mean(q_m), sqrt.(diag(cov(q_m))), exp.(-mean(q_s) ./ 2), mean(q_π))
row("conjugate VMP", mean.(result_conjugate.posteriors[:m]), std.(result_conjugate.posteriors[:m]), 1 ./ sqrt.(mean.(result_conjugate.posteriors[:w])), mean(result_conjugate.posteriors[:π]))

# %%
ys = range(minimum(y) - 1, maximum(y) + 1; length = 400)
density_laplace = [sum(mean(q_π)[k] * pdf(Normal(mean(q_m)[k], exp(-mean(q_s)[k] / 2)), v) for k in 1:K) for v in ys]
density_conjugate = [sum(mean(result_conjugate.posteriors[:π])[k] * pdf(Normal(mean(result_conjugate.posteriors[:m][k]), 1 / sqrt(mean(result_conjugate.posteriors[:w][k]))), v) for k in 1:K) for v in ys]

histogram(y; bins = 40, normalize = :pdf, color = :gray, alpha = 0.4, label = "data", xlabel = "y")
plot!(ys, density_laplace; color = :red, linewidth = 2, label = "Laplace + exact z")
plot!(ys, density_conjugate; color = :blue, linewidth = 2, linestyle = :dash, label = "conjugate VMP")

# %% [markdown]
# ## Plug-in vs expected messages
#
# How much does computing proper expectations matter? We rerun the model with the built-in plug-in fallback. To make it work at all, it must collapse a categorical marginal to its vector of probabilities rather than its mean, so we pass that as its `extract` function. With $N = 300$ observations the marginals of $m$ and $s$ are narrow and the two fallbacks agree closely; the difference grows as the data become scarcer, because the plug-in ignores the uncertainty of the other variables. In particular, it ignores $\mathrm{Var}[m_k]$ when it updates $s_k$, and so tends to underestimate the component spread.

# %%
plugin_extract(q) = q isa Categorical ? probs(q) : mean(q)

function fit(y, fallback)
    result = infer(
        model = laplace_mixture(K = K),
        data = (y = y,),
        constraints = laplace_constraints,
        initialization = laplace_initialization,
        iterations = n_iterations,
        returnvars = KeepLast(),
        options = (rulefallback = fallback,),
    )
    return exp.(-mean(result.posteriors[:s]) ./ 2)
end

for n in (300, 40, 15)
    expected_sds = fit(y[1:n], ExpectedLogPdfFallback())
    plugin_sds = fit(y[1:n], NodeFunctionRuleFallback(plugin_extract))
    println("N = ", lpad(n, 3), ":  component sd, expected = ", round.(expected_sds; digits = 3), ",  plug-in = ", round.(plugin_sds; digits = 3))
end

# %% [markdown]
# ## Exact vs approximate, and what did not work
#
# - **What worked.** Gaussian approximations for the continuous parameters, a Gaussian in log-space for the positive precisions, and exact sums for the discrete assignments together recover the mixture, and agree with RxInfer's hand-derived conjugate solution. No message update rule was derived by hand.
# - **What we had to build.** Discrete variables need their own projection (`CategoricalForm`), and a proper variational message needs an expectation over the other marginals (`ExpectedLogPdfFallback`). Both are generic, and both are prototypes of what the planned approximation layer will provide: the latter is precisely the `expectation(f, q)` operation that exact methods (closed forms) and approximate methods (quadrature, sampling) will implement.
# - **What did not work out of the box.**
#   - The built-in `NodeFunctionRuleFallback` with its default `extract = mean` fails on this model, because the mean of a categorical distribution is not a component index.
#   - The rule fallback does not apply to RxInfer's mixture nodes (`NormalMixture`, `Mixture`), because they have interface groups. We had to write the mixture likelihood with an explicit assignment variable.
#   - The free energy is not available: marginal rules and average energies have no fallback, and the fallback messages are unnormalised. We monitored convergence through the posterior means instead.
#   - The components must be initialised differently to break the symmetry of the mixture, as in any variational treatment of mixtures.
