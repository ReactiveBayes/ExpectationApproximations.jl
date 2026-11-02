# Plan: universal approximate inference

How we reach [VISION.md](VISION.md). This file is the single source of truth for scope,
order, decisions and status across all repositories involved. It lives on the
`universal-inference` branch of ExpectationApproximations.jl (EA), the hub.

Started 2026-10-29. Status is tracked in [Status](#status); decisions in
[Decisions](#decisions); everything still open in [Open questions](#open-questions).

## Goal of the first release wave

`infer(model, data)` runs a purely Laplace-based inference on non-conjugate models with no
hand-written rules, no per-node methods and no initialisation, demonstrated in
RxInferExamples.jl. EP, CVI and exact refinements follow on the same foundations.

Order of work: **BayesBase → ExpectationApproximations → engine hooks → universal rules →
RxInfer → RxInferExamples**, with ClosedFormExpectations and the migration of existing
packages in parallel.

## Milestones

Each milestone names its repositories, its deliverables and how we know it is done. IDs (`U2`,
`U2.3`) are used in branch descriptions, PR titles and commit messages.

### U0. Foundations and a spike

Repositories: EA (hub), scratch environments.

1. Check out every repository involved locally and set up
   the shared development environment ([Working across repositories](#working-across-repositories)).
2. Prototype the BayesBase protocol in a scratch environment: `expectation`, `supports_expectation`,
   labels, `with_expectation_rng`, `ExpectationFirstOf`, `ExpectationDynamicRouter`, with a toy
   closed-form backend and a Gauss-Hermite stub. Confirm with `@inferred`, JET and `@allocated` that
   `ExpectationFirstOf`'s choice and `has_expectation_label` fold at compile time and that
   `with_expectation_rng` does not allocate. Measure the cost of `ExpectationDynamicRouter` per
   call.
3. **Laplace spike in today's RxInfer**, before any refactor: `rulefallback =
   NodeFunctionRuleFallback()` plus a hand-written Laplace form constraint on every variable,
   on logistic regression and a hierarchical model with a Gamma precision. Goal: find every
   place the engine gets in the way (missing context in fallbacks, marginal rules, free energy,
   initialisation, constrained domains) with real models, and record them below.

Done when: the prototype's measurements and the spike's findings are recorded in
[Decisions](#decisions) and the milestones below are adjusted.

#### Findings of the Laplace spike (U0.3)

Three notebooks in RxInferExamples.jl, `examples/Universal Inference/` on branch
`universal-inference`, run against registered RxInfer 6.1.1 / ReactiveMP 7.0.1 /
MessagePassingRulesBase 1.1.3. All machinery lives in the notebooks: a `laplace(logp, x0)`
kernel (damped Newton, ForwardDiff), a `LaplaceForm` form constraint, fused stochastic nodes,
and in the mixture a `CategoricalForm` and an `ExpectedLogPdfFallback`.

| Model | Scheme | Result |
|---|---|---|
| Bioassay logistic regression (BDA3 §3.7) | BP, `q(θ) :: LaplaceForm()`, one pass | equals Laplace of the hand-written posterior to 0.0; vs exact grid: E[β] 7.2 vs 10.35, sd(β) 4.3 vs 4.9, 4.7% of mass on β < 0, LD50 90% interval too wide |
| Poisson log-linear regression | same, no new code but the node | recovers the rate within 1-2 sd |
| Stochastic volatility, T = 200 | BP, `μ(x)` and `q(x) :: LaplaceForm()` | linear-Gaussian check equals the Kalman smoother to 4e-16; RMSE of means vs slice sampler 0.18 (global Laplace 0.10), of sds 0.019 (0.011) |
| Gaussian mixture, K = 2, N = 300 | mean field, Laplace for m and log-precisions, exact z | matches the conjugate `NormalMixture` VMP to about 0.01 in every summary |

What got in the way, by milestone:

- **U3.** `NodeFunctionRuleFallback` collapses inputs with `mean`, which is wrong for a
  `Categorical` (the mean index, not the probabilities) and ignores input variances; the
  expectation-based fallback fixed both and visibly changes the component spreads at small N.
  The fallback does not apply to deterministic nodes (nonlinear functions had to be fused into
  stochastic nodes) nor to nodes with groups (`NormalMixture`, `Mixture`). No free energy.
- **U3/U5.** Under BP the form constraint must also be placed on messages (`μ(x) ::`), not
  only on marginals; a product without a Gaussian factor (a likelihood alone, e.g. at the end
  of a chain) needs a start point for the mode search.
- **U2.** Needed kernels: `laplace` (scalar and vector), Gauss–Hermite product rules over
  Gaussians, exact sums over categoricals: exactly the expectation protocol's first methods.
- **U7.** Projecting each message on its own is not Laplace propagation: the SV means trail the
  global Laplace solution, and iterating changes nothing. Cavity-based Laplace propagation needs
  the incoming message inside the projection.
- **Accuracy.** Laplace misses skew (bioassay β) and puts mass outside the data-supported
  region; EP is the natural comparison once available.

#### Experiment: automatic domain transforms (U0.3b)

Question: can constrained variables keep their natural priors (`p ~ Beta`, `λ ~ Gamma`) and be
handled by the projection alone, without rewriting the model in unconstrained coordinates?
Model: zero-inflated Poisson, `y[i] ~ ZeroInflatedPoisson(p, λ)` (a fused node), mean field.
`TransformedLaplaceForm` reads the domain from the support of the prior factor in the product
(`Distributions.support`), runs Laplace in log or logit coordinates with the log-Jacobian, and
returns a `LogNormal` or `LogitNormal`. Now the RxInferExamples notebook "Laplace Approximation:
Constrained Variables" (McElreath's drunk monks, N = 365, p = 0.2, λ = 1).

| Check | Result |
|---|---|
| Poisson–Gamma (conjugate, exact posterior known) | equals Laplace in log coordinates computed by hand, to rounding |
| ZIP, N = 100, p = 0.3 | E[p] 0.372 vs grid 0.369, sd 0.052 vs 0.053; E[λ] 2.72 vs 2.71, sd 0.227 vs 0.230; converges in 3 iterations |
| same, untransformed Laplace | also fine (0.366): far from the boundary the transform does not matter |
| ZIP, N = 10–15, p near 0 (three datasets) | transformed: runs, E[p] 0.17–0.18 vs grid 0.14 (skew again); untransformed: **fails**, Newton steps to p < 0 (`DomainError`) |
| notebook, N = 365, p = 0.2, λ = 1 (posterior corr(p, λ) = 0.74) | mean-field transformed: E[p] 0.166 vs grid 0.127, sd 0.046 vs 0.067; E[λ] 0.793 vs 0.762, sd 0.054 vs 0.076 |
| same, joint transformed Laplace over (logit p, log λ), computed directly | E[p] 0.166, sd 0.071; E[λ] 0.787, sd 0.079: the means carry the Laplace (skew) error, the mean-field sds the factorisation error |
| same, untransformed Laplace | **fails** even with a year of data: Newton overshoots λ < 0, and, fundamentally, quadrature over a Gaussian q(p) has nodes at p < 0 (4 of 16), still failing with a domain-safe line search |

Answer: **yes, it works without rewriting the model**, and the transform is what makes Laplace
valid near a boundary. What it needs, by milestone:

- **U2/U4.** Pushforward families have incomplete APIs: Distributions defines no `mean` or `var`
  for `LogitNormal`, BayesBase no `mean(log, ::LogNormal)` (exact: μ). Summaries and downstream
  rules must get these through the expectation protocol: exact closed forms from
  ClosedFormExpectations where they exist, quadrature in the unconstrained space otherwise.
  Conjugate rules that need only `E[τ]` already accept a `LogNormal` marginal (checked with
  `NormalMeanPrecision`); rules and energies that need `E[log τ]` would fail today.
- **U2.** At the first iteration the marginals are the priors (Beta, Gamma), so the default
  backend needs a rule for any univariate distribution; Gauss–Legendre in probability space
  (`quantile` at the nodes) worked.
- **U4.** Initialisation is required: without it the mean-field schedule never starts
  ("posteriors were never computed"). Initialising from the priors was enough.
- **U4.** Transforms per component of a vector variable. Strongly correlated parameters belong
  in one block (D15), and a block can mix domains (a probability and a rate here); the
  projection needs a product bijection with a summed log-Jacobian and a joint pushforward.
- **Open.** The domain comes from a factor with a known support in the product; a variable
  without such a factor (only deterministic or fused neighbours) has no domain yet.

#### Findings of the protocol prototype (U0.2)

Code, tests and measurements in `prototype/protocol/` on this branch: `ProtoBase` stands in for
the BayesBase additions, `ProtoBackends` for EA and ClosedFormExpectations. Written test first:
247 tests for `ProtoBase` on Julia 1.13 (225 on 1.10), 129 for `ProtoBackends`, Aqua with
ambiguity and piracy checks on. Details in `prototype/protocol/RESULTS.md`.

| Measurement | Result |
|---|---|
| Routing (`ExpectationFirstOf`) | decided by inference: `which_expectation_backend` compiles to `return <backend>`; JET clean |
| Labels | `is_expectation_exact` compiles to a constant, for backends, routers and products |
| `with_expectation_rng` | 0 bytes, nested routers included |
| Prepared path | 0 bytes for products of Categorical × MvNormal × PointMass, with the mixture notebook's integrand differentiated twice by ForwardDiff |
| `ExpectationDynamicRouter` | 1.4–1.8 µs and 1.4–2.3 KB per call, against 8.7 ns and 0 bytes for `ExpectationFirstOf`; freezing recovers the static numbers |
| Collisions | none with `using RxInfer`, since nothing is exported (D18) |
| Operation axis | can be added later, additively (D21) |
| Acceptance | the mixture and zero-inflated Poisson notebooks, their fallback rewritten with the protocol: posteriors equal to ≤ 1.4e-15; compiled inference time unchanged (6.04 → 6.18 s, 1.31 → 1.17 s) |

What the design had not said yet, now decided (D18–D22):

- A preparation is chosen without the integrand: `supports_expectation(backend, Any, Q)` means
  "every integrand". A prepared router prepares that backend and still sends an integrand with a
  closed form to its closed form, decided by inference.
- Numerical preparations are weighted point sets, the parts of product rules; products are
  routed component by component, and labels combine per label (`combine_expectation_label`).
- The default `expectation` goes through the preparation, guarded by `UnsupportedExpectationError`.
  Backends define methods for specific families only; routers define the product methods.
- Quantile quadrature over Beta and Gamma priors converges algebraically (about n⁻²): fine as
  the universal fallback, but U2's Gauss–Laguerre and Gauss–Jacobi rules matter.

### U1. BayesBase: the expectation protocol

Repository: BayesBase.jl. Additive, minor release.

Implemented from the prototype (`ProtoBase`, now in this branch's history) in
`src/expectations/` and `test/expectations/` of BayesBase, as release 1.7.0. Every name is
public and none is exported (D18). The prototype's backends, acceptance test and experiments
in `prototype/protocol/` run against it.

1. `expectation(backend, f, q)`, falling back to the preparation and guarded by
   `UnsupportedExpectationError`; `supports_expectation` at type level (hook) and instance level
   (API), with `supports_expectation(backend, Any, Q)` meaning "every integrand";
   `which_expectation_backend` for every backend.
2. Labels: `ExpectationLabel`, `ExpectationExact`, `ExpectationStochastic`; `has_expectation_label`
   per backend and per pair; `is_expectation_exact`, `is_expectation_stochastic`;
   `combine_expectation_label` for products (D20).
3. `with_expectation_rng`, `get_expectation_rng`; the integrand `Logpdf`, callable for anything with
   a `logpdf` (since replaced by `Base.Fix1(logpdf, p)`, D36).
4. Routers: `ExpectationFirstOf` with `prefer` and `exclude`; opt-in `ExpectationDynamicRouter` with
   `ExpectationBackendRegistry`, `register_expectation_backend!` and `which_expectation_backend`;
   `ExpectationFirstOf(ExpectationDynamicRouter())` freezes the choice.
5. No operation axis in this release (D21); it can be added later without breaking anything.
6. **Prepared expectations** (D13): `prepare_expectation(backend, q)` returns an object that
   evaluates `E_q[f]` for many `f` without rebuilding nodes or weights, and through which
   ForwardDiff numbers pass. The universal fallback evaluates `x ↦ E_q[log f(x, ·)]` inside a
   Newton loop, so this is the hot path, not an optimisation.
7. **Independent products** (D14): `FactorizedJoint` (D19), with components of different
   kinds (Categorical × MvNormal × PointMass), `supports_expectation` and labels composed from the
   components, routed component by component, computed as a product rule over point sets.
8. **Point sets** (D20): `ExpectationPointSet(points, weights)` and `ExpectationProductPointSet`,
   the preparation of every numerical backend, with a generic, dual-safe and allocation-free sum.

Done when: tests including inference and allocation tests, Aqua with ambiguity and piracy
checks, JET, and documentation of the protocol and of the label contracts.

### U2. ExpectationApproximations 1.0

Repository: EA. A rewrite; ideas from the old EA and from MessagePassingRulesApproximations
PR #2, new code.

1. **Lean core**: LinearAlgebra, Random, ForwardDiff, StaticArrays, Distributions, BayesBase.
   Heavy dependencies in package extensions. `using ExpectationApproximations` adds under
   0.2 s on top of Distributions.
2. **Kernels** (public, not exported, positional): Gauss-Hermite, Gauss-Laguerre,
   Gauss-Jacobi, Gauss-Legendre (Golub-Welsch, no FastGaussQuadrature), unscented and
   generalised unscented, spherical-radial cubature, exact sums over finite discrete
   distributions, product rules for mixed inputs, self-normalised importance sampling.
   Generic in the element type, so that ForwardDiff can differentiate through them: Laplace on
   expected log-densities depends on it.
3. **Backends** implementing the protocol with labels: one per method, plus `EADefault`
   (Gauss-Hermite for Normal, Gauss-Laguerre for Gamma, exact sum for finite discrete).
4. **Local Gaussian projections**, the operators every universal scheme uses:
   `laplace` (mode by damped Newton with AD, covariance from the Hessian, positive-definite
   correction), `tilted_mean_cov(backend, logf, q)` (moment matching), `mean_cov` and
   `mean_cov_cross` (pushforward moments), `linearize`.
5. **Domain transforms**: bijections to unconstrained space (log, logit, simplex) with
   log-Jacobians, chosen from a distribution's support; pushforward of the Gaussian back.
6. **Prepared path** (zero allocations for vectors and matrices) and the conformance suite
   `EA.Testing.test_backend`: correctness, labels, inference, allocations.
7. Constructors compatible with MessagePassingRulesApproximations 1.0, so its 1.1 can be a shim.

Done when: the conformance suite passes for every backend; zero allocations on scalars and
static arrays; CI with Aqua, JET on Julia 1.12+, benchmarks on pull requests.

#### What U2 delivered and found

The rewrite is on this branch, one tested commit per step (D22): 977 tests on Julia 1.13 (976 on
1.10), Aqua with every check, JET clean, every backend through the conformance suite.

| Item | Result |
|---|---|
| Lean core | BayesBase, Compat, Distributions, ForwardDiff, LinearAlgebra, Random, SpecialFunctions, StaticArrays, StatsFuns, Test; ExponentialFamily as an extension. `using` costs 0.21 s on top of Distributions, of which 0.03 s is this package and 0.14 s BayesBase with StaticArrays |
| Rules | Golub–Welsch Hermite, generalised Laguerre, Jacobi, Legendre, accurate to 1e-13 against 256-bit arithmetic; FastGaussQuadrature's smallest Hermite weights are off by up to 7e-10 at n = 50 |
| Backends | GaussHermite (Normal, MvNormal with static points, LogNormal, LogitNormal, TransformedGaussian), GaussLaguerre (any Gamma shape, the old rule was singular below one), GaussJacobi, QuantileLegendre, ExactSum, MonteCarlo, SNIS, Unscented, GenUnscented (exact to degree 4 for any univariate distribution), SphericalRadialCubature, and `EADefault` |
| Projections | `laplace` (in any domain, with `logZ`), `mean_cov`, `mean_cov_cross`, `tilted_mean_cov` (log space), `linearize` |
| Domains | RealLine, Positive, UnitInterval, LowerBounded, UpperBounded, Bounded, Simplex, Componentwise; `domain`, `forward`, `inverse`, `logjacobian`, `pushforward` |
| Conformance | `EA.Testing.conformance` / `test_backend`, itself tested on deliberately broken backends |
| MPRA compatibility | `EA.MPRACompat`, MPRA 1.0's API ported and checked against 1.0 itself |
| Acceptance | the mixture and zero-inflated Poisson notebooks on EA's backends: posteriors equal to ≤ 2.2e-15 |

What it found:

- **A BayesBase bug.** Prepared routers allocated on every evaluation: methods that only pass the
  integrand on were not specialised on it. Fixed in BayesBase (34ede71, for #69) with a test that
  checks every method, pushed to #69.
- **`E[log x]` under a Gamma converges slowly by quadrature** (1e-3 relative at 16 points, `log`
  being singular at zero): an expectation for closed forms (ClosedFormExpectations), which the
  router puts first.
- **Differentiating through a Gamma's or Beta's shape** is refused by Gauss–Laguerre and
  Gauss–Jacobi, whose nodes depend on it; a dual scale with constant shapes works.
- **MPRA 1.1 as a shim** (a local MPRA branch re-exporting `EA.MPRACompat`): MPRA's own suite passes
  but for its closure test, which forbids Distributions in its dependencies;
  DeltaMessagePassingRules, GCVMessagePassingRules, StandardMessagePassingRules,
  FlowMessagePassingRules (once the shim also binds the internal accessors `getα`, `getL`, … its
  tests use), PolyaMessagePassingRules and RxInfer's nonlinear and HGF tests pass.
- **Load time**: 0.2 s on top of Distributions is met by this package's own code, not by its
  dependencies; BayesBase's StaticArrays dominates.

### U3. Engine hooks: MessagePassingRulesBase and ReactiveMP

Repositories: MessagePassingRulesBase.jl, ReactiveMP.jl. Can run in parallel with U2.

1. **Fallbacks see what rules see.** Today a fallback is called as `fallback(fform, target,
   args)`: no context, no algorithm, no node instance, so it cannot reach `ctx.expectation`,
   the RNG, or a Delta node's function. Give it the algorithm and the `RuleContext`, keeping the
   three-argument form working for existing fallbacks.
2. **Fallbacks for marginal rules and average energies**, so joint marginals and the Bethe free
   energy work on the universal path.
3. **`:expectation` in `DEFAULT_CONTEXT_SERVICES`.** Rules read `ctx.expectation`; the engine
   always supplies it.
4. **A default projection for all random variables** (D15), composed where
   `EnsureSupportedFunctionalForm` is composed today, so "every variable Gaussian (in its
   unconstrained space)" can be one setting instead of one line per variable. It dispatches
   on the variable's domain (Gaussian, possibly in transformed coordinates, for continuous;
   exact normalisation for finite discrete), applies to **messages as well as marginals**
   (under BP a message leaving a variable is a product too), and returns a log-normaliser.
5. **Damping** as a stream postprocessor, needed by EP and loopy schedules later.
6. **Fallback coverage** found by the spike: deterministic (Delta) nodes and nodes with
   interface groups (mixtures) get no fallback today; the default fallback should take
   expectations over its inputs (via `ctx.expectation`) instead of collapsing them with `mean`.

Done when: existing behaviour unchanged (`NodeFunctionRuleFallback` tests pass as they are),
new hooks tested, documentation updated (fallbacks, context, form constraints).

#### What U3 delivered and found

A branch `universal-inference` in MessagePassingRulesBase (1.2.0), ReactiveMP (7.1.0) and RxInfer
(6.2.0), one tested commit per step, their `[sources]` pointing at each other by path until they
are pushed. MessagePassingRulesBase: 946 tests on Julia 1.13, 943 on 1.10. ReactiveMP: 15650 and
15646. RxInfer: 15064 on both. DeltaMessagePassingRules, StandardMessagePassingRules and
GCVMessagePassingRules pass against the branches, and so does the acceptance test.

| Item | Result |
|---|---|
| 1. Fallbacks see what rules see | `call_fallback(fallback, fform, target, algorithm, ctx, args)` and `call_energy_fallback(fallback, fform, algorithm, ctx, args)`, public in MessagePassingRulesBase; the default asks a three-argument fallback about messages only (D31) |
| 2. Marginals and average energies | `MarginalMapping` and the node's activation keep the fallback; a joint or an average energy without a rule asks it, and events name it as the rule that ran |
| 3. `:expectation` | in `DEFAULT_CONTEXT_SERVICES`; ReactiveMP supplies `nothing` unless the context sets one (D32) |
| 4. Default projection | the hook, not the projection (D33): ReactiveMP's `constrain_form_logscale` lets a constraint report the log-normaliser of its input; RxInfer's option `default_form_constraint` gives every variable without a constraint of its own one, for messages and marginals or apart, `(q = …, μ = …)`. The Laplace projection that dispatches on the domain is U4's |
| 5. Damping | `ReactiveMP.DampingPostprocessor(β)`: node messages mixed in natural parameters through `ReactiveMP.damp`, an extension on ExponentialFamily (D35) |
| 6. Fallback coverage | `NodeFunctionRuleFallback` serves Delta nodes (the message on `out` pulled back through the function towards an input; the function at the plugged-in inputs towards `out`), members of groups when the node defines its node function, and the average energy `-E_q[log f]` of a mean-field node through `ctx.expectation`. Its messages stay plug-in (D34) |
| End to end | the mixture notebook's model has a free energy, `EADefault()` agreeing with the notebook's quadrature to 3e-11, non-increasing after five iterations, the posteriors unchanged (`prototype/protocol/acceptance/free_energy.jl`); in RxInfer's tests a toy node's free energy equals its closed form, a model with `z := tanh(x)` and no Delta method runs through the fallback and a projection, and damped coordinate ascent reaches the undamped posterior |

What it found:

- **A fallback's own method was ambiguous** with defaults typed on the target, so
  `call_fallback(::MyFallback, node, target, …)`, the natural way to write one, failed. The default
  is one untyped method that dispatches on the target inside.
- **A product with no closed form had no log scale** before any constraint saw it, so a constraint
  that only reports its normaliser could never define one. A `ProductOf` now records its sides'
  log scales, the cause `:unnormalised_product`, and a reported normaliser completes them; an
  exact projection of non-closed normal densities reproduces the closed-form product's log scale.
- **BayesBase has no product of a point mass with a density.** The plug-in towards a Delta node's
  `out` gives a point mass, which a variable then cannot multiply with the next node's message;
  the RxInfer test asks for `x` only. U4's projection towards `out` removes the point mass; the
  product belongs in BayesBase (P).
- **`ProjectedTo` minimises KL(q ‖ p)**, mode-seeking: its posterior is compared with the same
  projection of the exact log-density, not with the exact moments.
- **On Julia 1.10**, `[sources]` is not read, so the branches resolve only in an environment that
  develops them; RxInfer's ignored manifest from 1.13 cannot be used by 1.10 either.
- **ReactiveMP's docs link** `call_fallback` and `call_energy_fallback` through
  MessagePassingRulesBase's published inventory, which has them only once 1.2 is released.

### U4. Universal Laplace rules

Repository: a new rules package (name open, see [Open questions](#open-questions)), depending
on MessagePassingRulesBase, BayesBase and EA.

1. **Stochastic nodes**, any node with a `nodefunction`: the message towards a target is the
   expected log-density `log m(x) = E_q(y)[log f(x, y)]` over the other inputs, computed through
   `ctx.expectation` (a plug-in value when an input is a point mass, an exact sum when it is
   categorical).
2. **Laplace projection of marginals and messages** in unconstrained coordinates: mode and
   Hessian of the sum of incoming log-densities plus the log-Jacobian; the domain from the
   prior's support, per component for vector variables. Warm starts from the previous mode; a start
   point for products without a Gaussian factor. The log-normaliser (Laplace's evidence
   approximation) is returned with the Gaussian and becomes the message's log scale.
3. **Exact projection of finite discrete variables**: normalise the product by evaluating it
   at every value (the spike's `CategoricalForm`).
4. **Deterministic nodes** (D16): a fallback that pulls the outgoing message back through the
   function, `log m(x) = E_q(y)[log m_out(f(x, y))]`, exact under BP for a single input; towards
   `out` a projection of the pushforward. `Linearization()` stays an option when every message
   is Gaussian; it cannot serve `Bernoulli(logistic(x))`, whose message to `p` is a Beta.
5. **Average energies** for the free energy on the universal path.
6. **Automatic initialisation** from the priors (Gaussian in unconstrained space).

Done when, with no rules, methods or initialisation written by the user (D17):
- **correctness**, tight: on linear-Gaussian models under BP the posterior equals the Kalman
  smoother's, means and variances (the spike reaches 4e-16 today); on any model the result
  equals the same approximation computed directly (a Laplace fit of the hand-written
  posterior, a global Laplace on a chain);
- **quality**, reported: the canonical models of [VISION.md](VISION.md#what-success-looks-like)
  run, and their gap to long NUTS runs (or exact grids) is published per method, with
  regression tolerances on that gap, not on agreement with NUTS;
- conjugate models never trigger a fallback (checked through rule annotations).

### U5. RxInfer integration

Repository: RxInfer.jl.

1. **One user-facing switch** for the universal scheme (keyword or option, name open) that
   sets the fallback, the default form constraint, the expectation backend and initialisation.
2. **The context**: `ctx.expectation = with_expectation_rng(ExpectationFirstOf(CFEBackend(),
   EADefault()), rng)` by default, overridable from the inference options and per node.
3. **An approximation report** on the inference result: which nodes ran exact rules, which ran
   fallbacks, with which projection and which backend labels (via annotations or callbacks).
4. **Error hints**: `RuleNotFoundError` and the Delta-node hint point to the universal switch.
5. **The default**: decide whether the universal scheme is on by default, after U6.

Done when: the canonical models run from `infer` with one switch; the report is documented;
existing tests unchanged.

### U6. RxInferExamples: universal inference

Repository: RxInferExamples.jl.

1. A new "Universal inference" section, starting with purely Laplace examples. Done in the
   spike: bioassay logistic regression, Poisson regression, stochastic volatility, Gaussian
   mixture, zero-inflated Poisson with Beta and Gamma priors (automatic domain transform).
   Next: a pendulum state-space model, a hierarchical model with a Gamma precision, a Poisson
   count time series. Each compared with an exact or
   sampling reference.
2. "Simple Nonlinear Node" rewritten without its hand-written rules.
3. Later: the same models under EP, and with exact refinements via ClosedFormExpectations.

Done when: the examples run in the examples' CI and show the approximation report.

### U7. Beyond Laplace

Repositories: the universal rules package, EA, ReactiveMP.

1. **Laplace propagation and EP**: belief-propagation structure with cavities (rules that read
   the message on their own edge, as Probit's EP rules already do), Gaussian division with
   positive-definite correction, damping. Same code, two projections: Laplace or moment
   matching through EA's quadrature. Acceptance: Laplace propagation reaches the global
   Laplace solution on nonlinear chains (the stochastic volatility notebook is the benchmark:
   local projection has RMSE 0.085 to it today), and EP improves on Laplace against the
   sampler.
2. **CVI and natural-gradient projection** through `expectation_gradient`;
   ExponentialFamilyProjection as the projection to families beyond the Gaussian.
3. **Exact refinements**: ClosedFormExpectations as the first backend in `ExpectationFirstOf`,
   showcased on Delta nodes computing `exp` or `log` against a log-density, `Base.Fix1(logpdf, p)`.

### P. Ecosystem migration (parallel track)

- **ClosedFormExpectations 1.0** (BIASlab, Mykola): `CFEBackend`, `supports_expectation`, the
  `ExpectationExact` label; `mean(ClosedFormExpectation(), f, q)` deprecated for one cycle;
  Williams-product gradients once the gradient operation is settled. Depends on BayesBase only. Its
  own `Logpdf` (callable only for a `Distribution`) is replaced by `Base.Fix1(logpdf, p)`, and its
  recursive calls pass `backend` on instead of constructing `ClosedFormExpectation()`.
- **MessagePassingRulesApproximations**: PR #2 exports its own `expectation(method, f,
  params...)`; it should extend `BayesBase.expectation` instead of owning a second function
  of the same name (no clash today only because BayesBase exports nothing, D18).
- **MessagePassingRulesApproximations PR #2** is superseded: its algorithms (Golub–Welsch
  Laguerre, Hermite grids, sigma-point weights, GenUnscented) are in EA 1.0; close it once U2 lands.
- **MessagePassingRulesApproximations 1.1** is verified as a shim re-exporting `EA.MPRACompat`
  (local branch `universal-inference`, see the U2 findings); it must also bind the internal
  accessors its users' tests import. Originally planned as a shim re-exporting EA's types with
  deprecations (`approximate` → `mean_cov`, `unscented_statistics` → `mean_cov_cross`,
  `approximate_meancov` → `tilted_mean_cov`, `local_linearization` → `linearize`); `smoothRTS`
  moves to DeltaMessagePassingRules; then archived, still registered.
- **Rule packages** (Delta, Flow, GCV, Standard, Polya, RxGP): use EA directly; rules that need
  an expectation as one step call `expectation(ctx.expectation, f, q)`.
- **ExponentialFamilyProjection**: its ClosedFormExpectations extension becomes a strategy
  holding any backend, once gradients are part of the protocol.

### Critical path

```
U0 ─► U1 ─► U2 ─────────► U4 ─► U5 ─► U6 ─► U7
        └─► U3 (parallel) ─┘
        └─► P  (parallel; CFE 1.0 after U1, MPRA 1.1 after U2)
```

## Decisions

Decided in the design discussion of October 2026; each can be revisited with a new entry.

- **D1. No type piracy.** Every method is defined on a type its package owns. Backend-less
  routing calls such as `expectation(::Base.Fix1{typeof(logpdf), <:Laplace}, ::Normal)` therefore
  cannot live in a backend package.
- **D2. Thin protocol in BayesBase.** BayesBase owns `expectation`, `supports_expectation`, labels,
  `with_expectation_rng` and the routers. No new dependencies. EA and ClosedFormExpectations
  implement it on their own backend types; there is no dependency between EA and CFE.
- **D3. Routing is a value.** `ExpectationFirstOf(backends...; prefer, exclude)` is itself a
  backend; candidates and preferences live in its type, so the choice is made at compile time.
  Several policies coexist (RxInfer's, ExponentialFamilyProjection's, a user's). Routers nest.
- **D4. Automatic choice is opt-in.** `ExpectationDynamicRouter` picks among loaded backends, ranked
  by labels, then priority, then name, never load order. Type unstable by design; never a
  default; never used inside library code; `ExpectationFirstOf(ExpectationDynamicRouter())` freezes
  it.
- **D5. Labels are traits, not a type hierarchy.** `ExpectationExact` (equals the expectation up to
  rounding) and `ExpectationStochastic` (depends on an RNG) in BayesBase, declared per backend or
  per pair, conservative (true only when guaranteed), forwarded by composite backends, tested by
  the conformance suite. Packages may define further labels.
- **D6. The RNG is a property of the backend.** Stochastic backends hold it; `with_expectation_rng`
  fills it from above, recursively, only where unset. Calls never pass an RNG.
- **D7. Rules read `ctx.expectation` and nothing else.** The engine resolves backend, RNG
  and per-node overrides before rules run. A method that *is* a node's algorithm (the
  Delta node's `Unscented()`) stays in the algorithm.
- **D8. `Logpdf` is a struct**, not an alias of `Base.Fix1{typeof(logpdf)}`, whose
  constructor would itself be piracy. Superseded by D36.
- **D9. Names follow Base style** (`haslabel`, `isexact`, `withrng`), tentatively. Superseded by
  D36.
- **D10. Exact first, at every level.** Exact rules before fallbacks; exact backends before
  approximate ones in the default router.
- **D11. The universal path is a fallback.** It never replaces an existing rule; it is
  consulted only when resolution finds none.
- **D12. Laplace first.** The first universal scheme is Laplace with local projection of
  messages and marginals, under whatever factorisation the model uses (BP or mean field): no
  message division, no damping needed. Laplace propagation and EP follow in U7. *(Revised after
  the spike: BP with projected messages already works and is exact on linear-Gaussian models.)*

Added after the Laplace spike (U0.3):

- **D13. Prepared expectations are part of the protocol.** BayesBase reserves
  `prepare_expectation(backend, q)`; EA implements it. Integrands are closures evaluated many
  times, with dual numbers, so the protocol must be generic in the element type of `f`'s output.
- **D14. Independent products of mixed kinds** are a first-class `q`: exact sums over
  categorical parts, quadrature over Gaussian parts, plug-in for point masses, labels composed.
- **D15. The default projection is domain-aware** (Gaussian in unconstrained coordinates, or
  exact for finite discrete), applies to messages and marginals, and returns a log-normaliser.
  "Every belief is Gaussian" means: Gaussian or finite categorical, possibly in transformed
  coordinates, possibly over a block of variables.
- **D16. Deterministic nodes pull messages back through the function** by default; linearisation
  and the unscented transform are opt-in methods for all-Gaussian neighbourhoods. Fusing a
  function into a stochastic node is the workaround until then.
- **D17. Correctness and quality are tested separately.** Tight tolerances against the same
  approximation computed another way; reported gaps, not tight agreement, against NUTS. Laplace
  is mode-seeking and will miss skew by design (bioassay: E[β] 30% low).

Added after the protocol prototype (U0.2):

- **D18. Protocol names are public, never exported.** They are declared with
  `Compat.@compat public ...` (the keyword exists from Julia 1.11; CI runs 1.10) and used
  qualified, `BayesBase.expectation`. No clash with `Stochastic` and `isstochastic` of
  MessagePassingRulesBase and ReactiveMP, nor with `expectation` of MPRA PR #2.
- **D19. `FactorizedJoint` is the independent product of D14.** BayesBase already defines it
  as "a joint distribution of independent random variables"; the integrand receives a tuple.
- **D20. Point sets are the common preparation of numerical backends.** Products are product
  rules over them, routed component by component; labels combine with `combine_expectation_label`
  (`ExpectationExact`: all parts, `ExpectationStochastic`: any part, other labels conservatively).
- **D21. No operation axis now.** A four-argument `supports_expectation(backend, op, F, Q)` added
  later forwards to the three-argument form for `expectation`, so it breaks nothing.
- **D22. Strict test-driven development.** Every function has tests written before it, and
  type stability, allocations and folding are tests, not observations. A function that is hard
  to test is a design signal.
- **D23. The PackageTemplate tooling everywhere.** Runic and TestItemRunner as in the
  template (see "Definition of done"), applied to BayesBase in the U1 pull request and to this
  repository. In BayesBase the new JET item found six possible errors in existing code, fixed
  test first in their own commit. Here the legacy suite (to be rewritten in U2) keeps its
  outcome: thousands of errors from API drift and no compat entries, now visible because the
  items run after Aqua instead of not at all.

Added in U2:

- **D24. EA's names are public, never exported**, as the protocol's (D18); `EA.MPRACompat` exports
  MPRA's names for the shim.
- **D25. ExponentialFamily is a weak dependency** of EA: its univariate Gaussians and
  `GammaShapeRate` through an extension; its multivariate Gaussians are `AbstractMvNormal`s.
- **D26. Rules in Float64, points from the parameters.** Golub–Welsch nodes and weights are
  computed once in Float64; the points of an expectation are typed from the distribution's
  parameters, so duals pass through parameters and integrands; static means give static points.
- **D27. Transformed Gaussians are EA's `TransformedGaussian`**, with `LogNormal` and
  `LogitNormal` for the positive half-line and the unit interval; this settles the open
  question on their representation.
- **D28. Sigma points keep their covariance weight** in a `SigmaPointSet`, which `mean_cov`
  applies; they do not support products or tilting, their weights can be negative.
- **D29. MPRA's 1.0 API is ported into EA**, not reimplemented on the backends: its shapes,
  warnings, errors and fields are what the rule packages rely on, and the port is checked
  against 1.0 itself.
- **D30. MPRA 1.1 may depend on BayesBase and Distributions**, through EA: its closure test is
  relaxed to forbid only the rule system and ExponentialFamily, and the whole suite passes against
  the shim. Load time is not a blocker unless it is really bad (0.21 s on top of Distributions is
  fine), and quadrature's slow `E[log x]` under a Gamma is accepted, closed forms being first.
- **D31. Fallbacks are asked through `call_fallback`**, positional, with the algorithm and the
  rule context, its default forwarding to the three-argument call for messages only; a
  fallback is never asked a question it was not written for.
- **D32. ReactiveMP stays backend-agnostic**: `ctx.expectation` is `nothing` unless the context
  sets one; RxInfer supplies `EADefault()` in U5.
- **D33. The projection is a hook plus an option**: a form constraint may report the
  log-normaliser of its input (`constrain_form_logscale`), and RxInfer composes a default form
  constraint where `EnsureSupportedFunctionalForm` is composed. A variable's own constraint
  replaces the default; ReactiveMP needs no option of its own, since constraints compose with `+`.
- **D34. `NodeFunctionRuleFallback`'s messages stay plug-in**; expectations over its inputs are
  U4's universal rules. Its average energy takes the expectation, through the context's backend.
- **D35. Damping is a stream postprocessor on node messages**, in natural parameters for one
  exponential family, anything else passing; the equality chain's triggers, marginals and
  scores are left alone, and RxInfer's `stream_postprocessors` carries it.
- **D36. Every protocol name says it is about expectations** (review of BayesBase#69):
  `supports_expectation`, `which_expectation_backend`, `with_expectation_rng`,
  `get_expectation_rng`, `ExpectationExact`, `ExpectationStochastic`, `has_expectation_label`,
  `combine_expectation_label`, `is_expectation_exact`, `is_expectation_stochastic`,
  `PreparedExpectation`, `UnsupportedExpectationError`, `ExpectationPointSet`,
  `ExpectationProductPointSet`, `ExpectationFirstOf`, `ExpectationDynamicRouter`,
  `ExpectationBackendRegistry`, `register_expectation_backend!`; `expectation`,
  `prepare_expectation` and `ExpectationLabel` were specific already. `Logpdf` is gone: nothing in
  it was about expectations, and the integrand of `E_q[log p]` is Julia's `Base.Fix1(logpdf, p)`,
  on which a closed form dispatches with no alias and no constructor of ours. Point sets stay apart
  from `SampleList`: a rule, with unnormalised and possibly negative weights, not a distribution.

The protocol, as decided (BayesBase):

```julia
function expectation end                                   # expectation(backend, f, q) = E_q[f]
function prepare_expectation end                           # prep = prepare_expectation(backend, q)
                                                           # expectation(prep, f), any f, dual-safe (D13)

supports_expectation(backend, ::Type, ::Type) = false      # hook for backend owners
supports_expectation(backend, f, q) = supports_expectation(backend, typeof(f), typeof(q))

abstract type ExpectationLabel end
struct ExpectationExact <: ExpectationLabel end
struct ExpectationStochastic <: ExpectationLabel end
has_expectation_label(backend, label) = false
has_expectation_label(backend, ::Type{F}, ::Type{Q}, label) where {F, Q} = has_expectation_label(backend, label)
has_expectation_label(backend, f, q, label) =
    supports_expectation(backend, f, q) && has_expectation_label(backend, typeof(f), typeof(q), label)
is_expectation_exact(backend, f, q) = has_expectation_label(backend, f, q, ExpectationExact())

with_expectation_rng(backend, rng) = backend               # filled from above, once
get_expectation_rng(backend) = nothing

# E_q[log p] dispatches on Julia's `Base.Fix1(logpdf, p)`, no type of its own (D36)

struct ExpectationFirstOf{B <: Tuple, P <: Tuple, X <: Tuple}  # a router is a backend
    backends::B; prefer::P; exclude::X
end
```

The full, tested version, with point sets, products and `ExpectationDynamicRouter`, is
`src/expectations/` in BayesBase (U1), documented on its "Expectations" page.

## Open questions

- **Gradients**: `expectation_gradient(backend, f, q, space)`; the parameter-space types live
  in ExponentialFamily, so BayesBase can only declare it with an untyped `space`.
- **`Differentiable` as a core label**, needed by CVI and natural-gradient schemes.
- **The universal rules package**: a new package (`UniversalMessagePassingRules.jl`?), part
  of EA as an extension on MessagePassingRulesBase, or part of DeltaMessagePassingRules.
- ~~**The fallback signature**~~: positional, through `call_fallback` (D31).
- ~~**Where the Laplace projection lives**~~: a form constraint reporting its normaliser through
  ReactiveMP's hook, composed by RxInfer's option (D33); the constraint itself is U4's.
- **Domains**: how a variable learns its support when several nodes connect to it, or when
  no neighbour carries a support (U0.3b found the prior factor is enough when there is one).
- **The user-facing switch**: a keyword of `infer`, an option, or a constraint-level wildcard.
  U3 adds the option `default_form_constraint`, which U5 can build on.
- **Default on or off** for the universal scheme in RxInfer (decide after U6).
- **Threads**: per-task RNGs for stochastic backends in parallel engines.
- **ClosedFormExpectations 1.0**: Mykola's agreement to adopt the protocol.

## Working across repositories

### Branches and pull requests

- The hub is EA's `universal-inference` branch: VISION.md, PLAN.md and the decisions.
- Every repository uses a branch of the same name for this effort, or `universal-inference/<topic>`
  for parallel pieces of work.
- Pull requests name the milestone (`U3.1: fallbacks receive the rule context`) and link this
  plan; they are merged into each repository's main in the order of the critical path.

### Local checkouts

| Repository | Local path |
|---|---|
| ExpectationApproximations.jl (hub) | `~/Projects/Julia/ExpectationApproximations.jl` |
| BayesBase.jl | `~/Projects/Julia/BayesBase.jl` (local branch `universal-inference`) |
| ClosedFormExpectations.jl | `~/Projects/BIASlab/ClosedFormExpectations.jl` |
| MessagePassingRulesBase.jl | `~/Projects/Julia/MessagePassingRulesBase.jl` |
| ReactiveMP.jl | `~/Projects/Julia/ReactiveMP.jl` |
| RxInfer.jl | `~/Projects/Julia/RxInfer.jl` |
| DeltaMessagePassingRules.jl | `~/Projects/Julia/DeltaMessagePassingRules.jl` |
| ExponentialFamilyProjection.jl | `~/Projects/Julia/ExponentialFamilyProjection.jl` |
| MessagePassingRulesApproximations.jl | `~/Projects/Julia/MessagePassingRulesApproximations.jl` |
| RxInferExamples.jl | `~/Projects/Julia/RxInferExamples.jl` |

### A shared development environment

One environment outside all repositories ties the local checkouts together with Julia's
`[sources]` (Julia 1.11+; we use 1.13):

```toml
# ~/Projects/Julia/universal-dev/Project.toml
[deps]
BayesBase = "b4ee3484-f114-42fe-b91c-797d54a0c67e"
ExpectationApproximations = "9cdca625-6a81-42ba-9f59-d0fb182fc17f"
ClosedFormExpectations = "70ff922c-62d4-418d-abfc-e284e489b734"
# MessagePassingRulesBase, ReactiveMP, RxInfer, ...

[sources]
BayesBase = {path = "../BayesBase.jl"}
ExpectationApproximations = {path = "../ExpectationApproximations.jl"}
ClosedFormExpectations = {path = "../../BIASlab/ClosedFormExpectations.jl"}
```

Set up in U0.1 with `Pkg.develop(path = ...)`, which on Julia 1.13 writes the `[sources]`
entries itself. It develops BayesBase, ClosedFormExpectations, MessagePassingRulesBase,
ReactiveMP and RxInfer. Today's EA is not part of it: its 28 dependencies only hold resolution
back, and it joins at U2.

- **Pitfall.** Pkg carries over the path entries in a developed package's own (git-ignored)
  `Manifest.toml`. RxInfer's local manifest dated from when the rule packages lived in
  `ReactiveMP.jl/lib/`, and broke resolution. It was moved to `universal-dev/stale/`.
  Check for stale path entries when a checkout has a local manifest.

For CI on unreleased branches, a repository's test environment may point at a branch of
another one, `BayesBase = {url = "https://github.com/ReactiveBayes/BayesBase.jl", rev =
"universal-inference"}`, until that release is tagged. Such entries are removed before tagging.

### Parallel sessions

- One working copy per repository (a git worktree when several pieces run at once), one Claude
  session per working copy.
- Every session starts by reading this file and VISION.md from the hub, and works on the
  milestone it was given.
- Progress is reported in pull requests; the [Status](#status) table below is updated on the
  hub only, so it never conflicts.

### Definition of done for any piece

Tests written before the code (D22); Aqua with ambiguity and piracy checks; JET on Julia
1.12+; allocation tests on hot paths; documentation; a CHANGELOG entry; and a status entry here.

Every repository follows the ReactiveBayes
[PackageTemplate](https://github.com/ReactiveBayes/PackageTemplate) (D23):

- **Formatting:** Runic. `make format` and `make check-format` run the latest Runic, and the
  `Format` workflow uses `fredrikekre/runic-action@v1`. A reformat is a commit of its own,
  listed in `.git-blame-ignore-revs`.
- **Tests:** TestItemRunner. Every `@testitem` carries tags by area, shared code is a
  `@testsnippet` or `@testmodule`, and `test/runtests.jl` selects with
  `make test test_args="tag:<t> name:<n> <path>"`. Items tagged `:slow` run in `make test`
  and CI but not in `make test-fast`; allocation tests are tagged `:alloc`.
- **Quality:** `test/quality_tests.jl`, tagged `:quality`, holds Aqua (with compat entries for
  test dependencies), JET (Julia 1.12+) and the doctests.
- **CI:** the template's `CI.yml` (Julia 1.10, 1.11, 1.12, 1 and pre, codecov with OIDC),
  `Format.yml`, `CompatHelper.yml`, `TagBot.yml`, `DocPreviewCleanup.yml` and Dependabot.

## Status

| ID | Repository | State | Pull request |
|---|---|---|---|
| U0 | EA (hub), scratch | done: U0.1 (`universal-dev`), U0.2 (`prototype/protocol/`), U0.3 (findings under U0) | |
| U1 | BayesBase | draft until everything is tested together (1.7.0), after the template tooling and the JET fixes, with the names of D36: 6600 tests on Julia 1.13, 6578 on 1.10; docs build | [BayesBase.jl#69](https://github.com/ReactiveBayes/BayesBase.jl/pull/69) |
| U2 | EA | done on this branch (977 tests); MPRA 1.1 shim verified on a local MPRA branch, kept local while EA is private | |
| U3 | MessagePassingRulesBase, ReactiveMP, RxInfer | drafts on the pushed `universal-inference` branches, each taking the others from theirs through `[sources]` (see "What U3 delivered and found") | [MessagePassingRulesBase.jl#16](https://github.com/ReactiveBayes/MessagePassingRulesBase.jl/pull/16), [ReactiveMP.jl#687](https://github.com/ReactiveBayes/ReactiveMP.jl/pull/687), [RxInfer.jl#710](https://github.com/ReactiveBayes/RxInfer.jl/pull/710) |
| U4 | universal rules (new) | not started | |
| U5 | RxInfer | not started | |
| U6 | RxInferExamples | started: four Laplace notebooks on `universal-inference` | |
| U7 | several | not started | |
| P | CFE, MPRA, rule packages, EFP | not started | |

## Appendix: the ecosystem as of October 2026

Facts from a survey of the sources, so that sessions do not need to repeat it. Paths are
relative to each repository; line numbers as of the survey.

**MessagePassingRulesBase**
- Rule fallbacks: an engine consults a fallback only when resolution returns `RuleNotFound`
  (`docs/src/fallbacks.md`). `NodeFunctionRuleFallback(extract = mean)` collapses every other
  input to a point and returns a `NodeFunctionLogPdf`, an unnormalised log-density with only
  `logpdf` and `insupport` (`src/fallback.jl:33-68`).
- `nodefunction(node)`: the log-density of a stochastic node without groups as a keyword
  function, `(; out, μ, v) -> logpdf(node(μ, v), out)`; defined by `@define_factor_node`
  (`src/nodes.jl:190-206`, `src/node_macro.jl:161-165`). Deterministic nodes and nodes with
  groups have none.
- `getnodefn(node, target)`: a deterministic node's function, implemented by the engine on its
  node object; reached by rules as `getnodefn(ctx.node, Target(:out))` (`src/nodes.jl:208-219`).
- Algorithms: `AbstractAlgorithm` selects rules and carries parameters, it is not an inference
  scheme; `DefaultAlgorithm` (the factorisation decides BP, VMP or structured VMP);
  `DefaultAlgorithmExtension` overrides some rules and inherits the rest
  (`docs/src/algorithms.md`).
- `RuleContext`: services as a typed `NamedTuple`, read as `ctx.name`, `nothing` when
  missing; `DEFAULT_CONTEXT_SERVICES = (:node, :rng, :matrix_correction)`; `check_services`
  refuses a rule whose declared services are missing (`src/context.jl`, `src/rulespec.jl:673`).
- EP-style rules read the message on their own edge through dependency declarations
  (`src/dependencies.jl:150-198`); Probit in StandardMessagePassingRules works this way.

**ReactiveMP**
- Resolution: `MessageMapping` calls `find_message_rule`; on `RuleNotFound` with a
  `rulefallback` it calls `fallback_message(...)`, which calls the fallback as `fallback(fform,
  target, args)`; `nothing` means `RuleNotFoundError` (`src/message.jl:785-853`).
- The fallback gets no context, no algorithm and no node instance; marginal rules and average
  energies have no fallback (`src/marginal.jl:270-272, 373-383`, `src/score/node.jl:150-160`).
  Fallback messages have an undefined log scale.
- Context: `node_context` builds `RuleContext(merge((node, rng = default_rng(),
  matrix_correction = nothing), context))`, one per node, from the activation option `context`
  (`src/context.jl:30-34`, `src/nodes/nodes.jl:625`).
- Products and projections: `MessageProductContext` with `prod_constraint`, `form_constraint`
  and the check strategy; `constrain_form(constraint, ::ProductOf)` is the projection hook;
  `prepare_context` precomputes state (`src/message.jl:190-248, 296, 440`,
  `src/constraints/form.jl:145-152`).
- No cavity distributions, no message division, no damping. Stream postprocessors
  (`src/postprocessors.jl`) and callbacks exist. v6's Laplace, CVI, ProdCVI and quadrature
  were removed; Delta offers `Unscented`, `Linearization` and `CVIProjection`.

**RxInfer (6.1.0)**
- `infer` keywords include `constraints`, `algorithm` (formerly `meta`), `initialization`,
  `options`, `iterations`, `free_energy`; `options` accepts `context` and `rulefallback` among
  others, passed to every node (`src/inference/inference.jl:646-680`,
  `src/model/plugins/reactivemp_inference.jl:42-52, 111-180, 715-730`).
- No global default algorithm and no wildcard form constraint; every random variable gets the
  user's form constraint composed with `EnsureSupportedFunctionalForm`
  (`reactivemp_inference.jl:419-446`).
- Any undeclared function becomes `DeltaFn{F}`; with no algorithm there is no rule, and the
  error suggests `Linearization()`, `Unscented()` or `CVIProjection()`
  (`DeltaMessagePassingRules.jl/src/node.jl:25, 39-41, 122-125`).
- `ProjectedTo` lives in `ext/ProjectionExt/ProjectionExt.jl:11-66`. Free energy:
  `free_energy = true`; early stopping: `StopEarlyIterationStrategy`. No damping.

**RxInferExamples**
- Sections: Basic, Advanced, Problem Specific, Experimental. Linearization in 7 examples,
  Unscented in 6, CVIProjection in 3, `ProjectedTo` in 1; 14 examples ship hand-written rules
  (among them Simple Nonlinear Node, Ising Model, RS-LDS, VAEs). None uses Laplace or
  `rulefallback`.
