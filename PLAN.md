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

Repository: DeltaMessagePassingRules (D37), which gains EA, BayesBase 1.7 and
MessagePassingRulesBase 1.2 as dependencies; the form constraint's ReactiveMP methods are
RxInfer's.

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
6. ~~**Automatic initialisation** from the priors~~: U5's, with the switch (D38).

Done when, with no rules or methods written by the user (D17):
- **correctness**, tight: on linear-Gaussian models under BP the posterior equals the Kalman
  smoother's, means and variances (the spike reaches 4e-16 today); on any model the result
  equals the same approximation computed directly (a Laplace fit of the hand-written
  posterior, a global Laplace on a chain);
- **quality**, reported: the canonical models of [VISION.md](VISION.md#what-success-looks-like)
  run, and their gap to long NUTS runs (or exact grids) is published per method, with
  regression tolerances on that gap, not on agreement with NUTS;
- conjugate models never trigger a fallback (checked through rule annotations).

#### What U4 delivered and found

A branch `universal-inference` in DeltaMessagePassingRules (1.1.0), on top of the U3 branches, one
tested commit per step, with the docs of DeltaMessagePassingRules and RxInfer building:

| Repository | Julia 1.13 | Julia 1.10 |
|---|---|---|
| DeltaMessagePassingRules | 678 | 677 |
| EA | 1002 | 1001 |
| MessagePassingRulesBase | 953 | 946, and the registry fixture's error: its subprocess environment cannot resolve the unregistered BayesBase 1.7 without `[sources]` |
| ReactiveMP | 15654 | 15650 (4 version-gated tests fewer, as in U3) |
| RxInfer | 15095, the 31 acceptance items included | 15095 |
| StandardMessagePassingRules, GCVMessagePassingRules | 10193, 463 | |

| Item | Result |
|---|---|
| 1. Stochastic nodes | `ExpectationRuleFallback`: towards an interface or a member of a group, `x ↦ E_q[log f(x, ·)]` over every other input, prepared once through `ctx.expectation`, evaluated without allocating. Towards `out` of a node that is a distribution saying its support, a `ContinuousUnivariateLogPdf` on it; elsewhere a `NodeFunctionLogPdf` (D40) |
| 2. Laplace projection | `LaplaceForm` and `laplace_form(form, density, previous)`: closed forms pass, `EA.laplace` in `EA.domain(product)` otherwise, reporting `logZ`; a warm start from the variable's last mode (RxInfer's context); a point mass among the factors is the product |
| 3. Finite discrete | normalised exactly over the support, `logZ = logsumexp` |
| 4. Deterministic nodes | towards `out` the moment-matched pushforward, `EA.mean_cov` through `ctx.expectation`, a Gaussian (a point mass for point masses), so U3's point mass that nothing multiplies is gone; towards an input the pull-back, in expectation over the other inputs, exact for one |
| 5. Average energies | `-E_q[log f]` with joint keys and groups; joints of a cluster by joint Laplace (D42), so the Bethe free energy is available under BP; a Delta node's term is the entropy of the joint over its inputs |
| Log scales | `with_logscale` from a fallback, MessagePassingRulesBase's `unwrap_fallback`, ReactiveMP unwrapping it (D39): the bioassay's posterior log scale equals Laplace's evidence to 1e-8 |
| 6. Initialisation | moved to U5 (D38) |

Acceptance (RxInfer `test/inference/universal_laplace_tests.jl`), with only `rulefallback =
ExpectationRuleFallback()`, `default_form_constraint = LaplaceForm()` and an expectation backend:

| Model | Check | Result |
|---|---|---|
| linear-Gaussian chain under BP, observations through a node with no rules | Kalman smoother | posteriors and Bethe free energy equal to rounding |
| bioassay logistic regression | `EA.laplace` of the hand-written posterior | mean, covariance and log evidence to 1e-8 |
| Poisson regression | the same | to 1e-8 |
| `z := tanh(x)`, every variable returned | Laplace of the exact posterior | to 1e-8; z a Gaussian |
| `z := a * exp(b)` under BP | free energy | finite |
| conjugate iid Normal–Gamma (mean field) and Beta–Bernoulli | fallback events, posteriors, free energy | no fallback; identical with and without the options |
| zero-inflated Poisson, Beta and Gamma priors, mean field | grid | E[p] 0.441 vs 0.389, E[λ] 0.503 vs 0.475: within half a posterior sd (guarded); the sds halved by mean field |
| stochastic volatility, T = 50 | global Laplace | RMSE of the means 0.19 (guarded at 0.25) |

On the hub, `prototype/protocol/acceptance/universal_laplace.jl`: the mixture and zero-inflated
Poisson notebooks with the package pieces in place of their own give the notebooks' posteriors,
to 1.4e-14 and 1.3e-9 (Laplace kernels that stop differently); `compare.jl` and
`free_energy.jl` are unchanged.

What it found:

- **Under BP the expected log-density is not the BP message** unless every other input is a point
  mass (a likelihood of data) or, for a Delta node, there is one input; elsewhere it is the
  VMP-like message, so BP with fallbacks between two random variables is approximate. Cavity
  methods (U7) are the fix; the Kalman check holds because the transitions have their rules.
- **`mean(log, q)` exists for neither `LogNormal` nor `LogitNormal`** (ExponentialFamily, beside its
  `mean(log, Gamma)`, is where it belongs: P), so a Gamma prior's average energy failed on a
  Laplace posterior. Every constrained domain now pushes forward to EA's `TransformedGaussian`,
  which has `mean(f, q)` (D41).
- **Distributions gives no default `minimum`, `maximum` or `mean`** for a user's distribution: the
  fallback and the projection do without them, and a prior gives its domain only by saying its
  support (the RxInfer docs example adds `minimum` and `maximum`).
- **A likelihood's message alone** (the backward message at the end of a chain) tells no
  dimension; as in the spike it is projected as a density of a number when it takes one.
- **Pkg takes no unregistered test-only dependency from `[sources]`**, so RxInfer's tests load EA
  as DeltaMessagePassingRules' dependency, `Base.require(Base.PkgId(uuid, name))`; Pkg does read
  a path-sourced dependency's own `[sources]` (EA through DeltaMessagePassingRules).
- **RxInfer's docs environment** takes its direct dependencies from General, so it has not resolved
  against the U3 branches either; it gains the same temporary `[sources]`, and its stale manifest
  (from `ReactiveMP.jl/lib/`) was set aside.
- **EA is private**: a pushed branch can name it by its SSH URL, which Pkg resolves for anyone with
  access (checked); the HTTPS URL waits for credentials.
- **On Julia 1.10 a node's keyword function allocates** 16 bytes a call, so the prepared
  message's allocation check runs from 1.11.
- **Joints are of numbers**: a cluster with a vector member, or an input that is itself a joint
  (structured factorisations), gets no fallback yet.

### U5. RxInfer integration

Repositories: RxInfer.jl, with EA and DeltaMessagePassingRules for what they need.

1. **One user-facing switch**: `infer(...; fallback = LaplaceApproximation())` (D43), a
   `FallbackScheme` that sets the rule fallback, the default form constraint, the expectation
   backend and the initialization strategy, each unless `options` or `initialization` set it
   (D47).
2. **Automatic initialisation** (D38, D45): `initialization` takes a strategy as well as a
   specification. `FromPriors()` starts every random variable a specification leaves out from its
   prior at its parents' initial means; `FromPriors(@initialization …)` applies the block first.
   Priors cannot break a mixture's symmetry: its means are given in the block.
3. **The context** (D44): the scheme's default backend is the session's, the backends registered
   with BayesBase (EA registers `EADefault`'s rules when it loads), frozen once per call of
   `infer` into a type-stable `ExpectationFirstOf`, before the options reach the model;
   `LaplaceApproximation(expectation = …)` pins one. Per-node overrides are deferred (RxInfer
   has no per-node context).
4. **The approximation report's foundation** (D46): `infer(...; report = true)` and
   `result.report`, a composite `InferenceReport` whose section `approximations` records from the
   engine's events which nodes ran exact rules and which the fallback, the backend each input of
   the fallback went to, and each variable's projections with their outcome. The report as a
   product (its presentation and diagnostics) is U8.
5. **Error hints**: a missing rule's hint and RxInfer's guide name `fallback =
   LaplaceApproximation()`, the Delta node's hint too; variables never updated name
   `initialization = FromPriors()`.
6. **The default**: decide whether the universal scheme is on by default, after U6.

Done when: the canonical models run from `infer` with one switch; the report is documented;
existing tests unchanged.

#### What U5 delivered and found

On the `universal-inference` branches, one tested commit per step: EA's backend registration,
DeltaMessagePassingRules' projection record (`laplace_form` says its `method`) and Delta hint, and
in RxInfer `FromPriors`, `fallback`, `report` and the hints. The docs of RxInfer build.

| Repository | Julia 1.13 | Julia 1.10 |
|---|---|---|
| EA | 1016 | 1015 |
| DeltaMessagePassingRules | 687 | 686 (the allocation check runs from 1.11) |
| RxInfer | 15184, U5's 89 new tests included | 15184 |

| Item | Result |
|---|---|
| 1. The switch | `fallback = LaplaceApproximation(; tol, maxiters, expectation, initialization)`, through batch and streaming inference; `RxInfer.FallbackScheme` with `fallback_rule`, `fallback_form`, `fallback_expectation`, `fallback_initialization` for other schemes (U7's) |
| 2. Initialisation | `RxInfer.InitializationStrategy`, `RxInfer.initial_marginal(strategy, model, label)`; `FromPriors` walks each variable's `out` factor, constants by value and random parents by their initial means, recursively, deterministic parents as `f` at their inputs' means; only marginals are set |
| 3. The context | `register_backends!` in EA's `__init__`; `ExpectationFirstOf(ExpectationDynamicRouter())` with `with_expectation_rng` in `apply_fallback`; every node's rule context is concretely typed (tested) |
| 4. The report | `InferenceReport`, `ApproximationsReport` with `NodeRecord`, `VariableRecord`, `ProjectionRecord`; `RxInfer.form_outcome` (`LaplaceForm`'s context keeps its last `method`, `logZ`, `converged`, `iterations`); `report = (:approximations,)` selects sections; events it does not record are never built |
| 5. Hints | in RxInfer's log text, the never-updated error, the Rule Not Found guide and the Delta hint |

Acceptance: every U4 acceptance model, rerun with `fallback = LaplaceApproximation()` and no
options, gives the options' posteriors, free energies and log scales exactly (`==`), the
zero-inflated Poisson with no `initialization` (its priors are the hand-written initial marginals)
and the conjugate models with no fallback call. On the hub, `universal_laplace.jl` runs the two
notebooks through the switch: the zero-inflated Poisson with no initialization matches the
notebook to 1.3e-9, the mixture with its means alone given to 4.5e-13; `compare.jl` and
`free_energy.jl` unchanged.

What it found:

- **The frozen router orders backends of equal priority by name** (`ExactSum, GaussHermite,
  GaussJacobi, GaussLaguerre, QuantileLegendre`); EA's rules have disjoint supports, so it chooses
  as `EADefault` does, and the catch-all `QuantileLegendre` is registered last by priority.
- **Registering again needs equal backends**: `GaussHermite` and `QuantileLegendre` hold vectors,
  so they gained `==` and `hash`; `ExpectationFirstOf` has no `==` (tests compare by identity).
- **Initial marginals under belief propagation change nothing**: the switch's results equal the
  options' exactly although `FromPriors` initialises every variable.
- **Streaming models keep an `initialization`**: their priors' parameters are data (the
  autoupdates), which `FromPriors` cannot read when the model is built.
- **What the events do not say**: a deterministic node's free-energy term has no rule, and a
  joint Laplace drops its convergence (`expectation_joint.jl`), so neither is in the report yet;
  nodes are recorded by type, not instance. U8.

### U6. RxInferExamples: universal inference

Repository: RxInferExamples.jl.

1. A new "Universal inference" section, starting with purely Laplace examples. Done in the
   spike: bioassay logistic regression, Poisson regression, stochastic volatility, Gaussian
   mixture, zero-inflated Poisson with Beta and Gamma priors (automatic domain transform).
   **Moved to the U5 API** (see "The notebooks on the switch" below): each runs through
   `fallback = LaplaceApproximation()` with no machinery of its own, and hidden checks make it a
   test. Next: a pendulum state-space model, a hierarchical model with a Gamma precision, a
   Poisson count time series. Each compared with an exact or sampling reference.
2. "Simple Nonlinear Node" keeps its hand-written rules: it teaches writing a node and its rules,
   which stays worth teaching (the user's call). Its story is improved, and it links to the
   Universal Inference notebooks for the same model without rules.
3. Later: the same models under EP, and with exact refinements via ClosedFormExpectations.
4. A section on the inference report: how to enable it (`report = true`) and read it. Waits for
   U8, whose item 5 updates the notebooks to the report.

Done when: the examples run in the examples' CI and show the approximation report.

#### The notebooks on the switch

The four notebooks of `examples/Universal Inference/` on RxInferExamples' `universal-inference`
branch, one commit each:
- the notebooks' own Laplace kernels, form constraints, rule fallbacks, domains and moments
  (40 to 150 lines each) are gone; each calls `infer(...; fallback = LaplaceApproximation())`;
- the hand-written references call `ExpectationApproximations.laplace`;
- the report is not shown yet (U8, item 5).

**The notebooks are tests.** Lines ending in `# hide` run but are left out of the page.
Hidden `@assert`s guard two kinds of claim:
- **identities**: message passing equals the Laplace fit of the hand-written posterior, and the
  linear-Gaussian chain equals the Kalman smoother, free energy included;
- **numbers the text states**: β about a third low, the stochastic volatility RMSEs, the mixture
  against the conjugate `NormalMixture`, the mean-field standard deviations against the grid,
  and that a conjugate model under the switch never calls the fallback.

The examples' build missed an error that followed a cell's printed output, so a failing hidden
check passed. `has_error_blocks` in `examples/make.jl` now looks at every line of an output
block (its own commit, checked with a deliberately failing assert). The acceptance command,
which builds the four in the dev environment while the packages are unreleased:

```
make examples-env ENVIRONMENT=~/Projects/Julia/universal-dev FILTER=laplace
```

All four pass in 2 min 35 s: the mixture takes 146 s (seven fits of 30 iterations), the others
22 to 42 s. Their numbers are the spike's, plus some new ones:
- bioassay: −log p(y) is 5.30 on the grid and 5.36 by Laplace; the free energy is 6.41, an
  upper bound that the Gaussian misses by its KL divergence of 1.1 nats;
- stochastic volatility: RMSE of the means against the sampler 0.177, global Laplace 0.10;
- zero-inflated Poisson: E[p] 0.166 against the grid's 0.127.

A full `make examples` takes registered RxInfer, so these four fail there until the release.

**Found while moving them**, to settle before U7 (brainstorm with the user):
1. **`dot` in a model is RxInfer's `dot` node**, a declared deterministic node with exact
   Gaussian rules, and it shares the one rule fallback with Delta nodes (D48). The first record
   here, that `dot` throws, was wrong: the `iterate(::Normal)` is Bernoulli's energy (item 2). A
   callback trace of `p[i] := logistic(dot([1, x[i]], θ)); y[i] ~ Bernoulli(p[i])` without the
   free energy shows a silent wrong answer instead, the prior (E[θ] ≈ 0 against Laplace's
   (1.71, 11.80)), from three gaps:
   - **a declared function node carries no function**: RxInfer gives `nodefn` only to an
     undeclared function, so the fallback reaches `dot` and stops ("created without `nodefn`");
   - **a projection that found no mode is used**: Laplace on the lone pulled-back Bernoulli
     message, monotone in η, does not converge, the curvature floor gives σ = 10⁴, and `dot`'s
     exact rule turns that into an uninformative message (D49);
   - **two `ProductOf`s cannot be multiplied** (item 3), once the messages towards θ stay
     log-densities.

   See "One fallback for every deterministic node" below.
2. **A Delta node's forward message ignores the function's range.** For
   `p := f(θ, x)` with `f` a logistic, the message to `p` is a moment-matched Gaussian on the
   real line. Bernoulli's energy rule then calls `mean(log, ::Normal)`. **After the merge** (the
   user's call): outside this effort's scope.
3. **The same model without the free energy** stops at BayesBase:
   `prod(::GenericProd, ::ProductOf, ::ProductOf)` is ambiguous (20 pulled-back messages towards θ).
4. **`Binomial` is not a factor node**, and GraphPPL does not index a multivariate variable
   (`θ[1]`). So the bioassay keeps its likelihood node. Indexing a vector variable is a large
   change of GraphPPL's: **outside this effort** (the user's call).
5. **`logscales = true` returns posteriors as ReactiveMP `Marginal`s**, on which `rand` failed.
   **Fixed**: a `Marginal` samples as its distribution (ReactiveMP), and the bioassay keeps the
   evidence from its one `infer` call.
6. **The mixture has no free energy.** The energy rule of `m ~ MvNormalMeanCovariance(zeros(K),
   100I)` (StandardMessagePassingRules, `mv_normal_mean_covariance.jl:9`) calls `ndims` on the
   projection's Distributions `FullNormal`. Should `laplace_form` return ExponentialFamily's
   Gaussian types, or the rules accept any `MvNormal`? The bioassay's `θ ~ prior` does not hit
   this. **Fixed** by the former (D50): the mixture notebook plots its free energy, which falls
   from 670.8 to 554.16 in eight iterations and, once settled, rises by at most 3e-6.

#### One fallback for every deterministic node

From the brainstorm of finding 1 (D48, D49), one tested commit each, local on the
`universal-inference` branches:
1. **ReactiveMP**: `getnodefn` gives a node created without `nodefn` its form when that is a
   function. A test checks that `+` (a group), `-`, `*` (`A`, `in`) and `dot` take their inputs in
   their declared order, whatever order they are given in. 15663 tests.
2. **DeltaMessagePassingRules**: `LaplaceForm(nomode = :leave)` leaves a product Newton finds no
   mode for as it is, method `:unprojected`; `:project` stays the default. 692 tests.
   **RxInfer**: `LaplaceApproximation` projects marginals with `:project` and messages with
   `:leave`, `(q = LaplaceForm(), μ = LaplaceForm(nomode = :leave))`. 15184 tests, as in U5.
3. **BayesBase** (U1's branch): `prod(GenericProd(), ::ProductOf, ::ProductOf)` joins the right
   tree to the left one side at a time. 6619 tests.
4. **Acceptance** on the hub, `prototype/protocol/acceptance/deterministic_nodes.jl`. The
   bioassay as 20 Bernoulli trials, written `p := logistic(dot([1, x], θ))` and
   `p := dose_response(θ, x)` (a Delta node), equals the fused node and `EA.laplace` of the joint
   to 1.4e-9. Its trace: `dot → in2` through the fallback, its message on `out` being a
   log-density; `dot → out` by its own exact rule, its inputs Gaussian. `universal_laplace.jl`
   and `free_energy.jl` still pass.
5. **The logistic regression notebook** has a section "The model as it reads on paper" with the
   `dot` form, its posterior equal to the fused node's (hidden check). The four notebooks build
   in 2 min 34 s.

Before them, the `dot` form ran and returned the prior. Laplace on the lone pulled-back Bernoulli
message gave `Normal(±22.9, 10⁴)`, which `dot`'s exact rule took. Item 2 of the findings, the
range of a Delta node's output, keeps that form without a free energy; it waits until after the
merge.

### U7. Beyond Laplace

Repositories: DeltaMessagePassingRules, EA, ExponentialFamilyProjection, RxInfer; ReactiveMP for
U7a. Split into three, after the brief of November 2026:

#### U7a. Cavity methods: Laplace propagation and EP (on hold)

On hold, the user's call: rules that read the message on their own edge need more thought before
any design. Every method that needs a cavity waits with them. What the brief found, for when it
resumes:
- **The target.** A node without rules would send a Gaussian site,
  `proj(cavity · f) / cavity`. Laplace's method gives Laplace propagation, whose fixed point on
  a chain like stochastic volatility is the global Laplace approximation, means and covariances.
  Moment matching through EA's `tilted_mean_cov` gives EP.
- **Own-edge rules work in ReactiveMP.** Probit's EP rule declares `:in => (m[:out], m[:in])`
  with an initial message (StandardMessagePassingRules `src/nodes/probit.jl:72-82`). ReactiveMP
  iterates such cycles to the fixed point: two Probit nodes feeding each other's cavity reach
  hand-computed EP to 1e-10 in ten iterations (`test/engine/approximation_tests.jl:156`).
- **A fallback never sees its own edge.** Its arguments are the node's dependency scheme, which
  leaves the target out (ReactiveMP `src/nodes/dependencies.jl:127-143`), and adding `m[:x]`
  everywhere would stop exact rules from matching. The cavity could be added only where a node
  has no rule for the target (`list_rules`), or at the moment a fallback is asked, a deeper
  engine change. Neither is decided.
- **There is no global per-algorithm switch** in RxInfer, and `FallbackScheme` has no hook for
  damping or initial messages.
- **Sites must be eager Gaussians.** DMPR's `DivisionOf` cancels only on two-connection
  variables. Division would be a natural-parameter subtraction, as Probit does, with a
  positive-definite correction, because `damp` errors on improper natural parameters.
- **Names chosen:** `LaplacePropagation()` and `ExpectationPropagation()`, as fallback schemes
  (D43).

#### U7b. CVI and natural-gradient projection (this round)

CVI as a projection, with no cavity (D51).
- Under mean field a variable's marginal is the product of its messages. Minimising KL(q‖p̃) of
  that product over Gaussians gives the optimal mean-field Gaussian, the CVI fixed point.
- The natural-gradient step in mean parameters needs E_q[∇ log p̃] and E_q[∇² log p̃]
  (Bonnet/Price) through EA's quadrature (D52). Gradients of an expectation are not needed.
- Items:
  1. EA `gaussian_variational(logp, x0[, domain]; backend, …)`, Laplace's sibling, returning the
     ELBO as `logZ`.
  2. DMPR `VariationalForm`, `LaplaceForm`'s sibling.
  3. RxInfer `fallback = CVI()`.
  4. ExponentialFamilyProjection: a deterministic strategy holding a BayesBase expectation
     backend, `ExpectationStrategy(backend)` (D53), so that `ProjectedTo(Beta)` and other
     families run on quadrature.
  5. The notebooks: CVI against Laplace on the bioassay; a Beta posterior for the zero-inflated
     Poisson's two-week case.

Acceptance:
- `CVI()` equals `gaussian_variational` of the hand-written joint on the bioassay, and its free
  energy is at most Laplace's there and under mean field on the zero-inflated Poisson;
- the linear-Gaussian chain is exact;
- a Beta projection of a Beta-Bernoulli posterior through a rule-free likelihood equals the
  conjugate Beta.

#### What U7b delivered and found

Local on the `universal-inference` branches, one tested commit per step, not pushed;
ExponentialFamilyProjection gains a `universal-inference` branch (3.5.0).

| Item | Result |
|---|---|
| 1. EA `gaussian_variational` | `gaussian_variational(logp, x0[, domain]; backend, tol, maxiters) -> (; distribution, logZ, mean, iterations, converged)`, `logZ` the evidence lower bound (D52). 5 Newton steps on a skewed 1-D density, 6 on the bioassay; exact in one for a Gaussian. Tested against the fixed-point conditions with a 64- or 40-point quadrature and against Laplace's bound. 1046 tests. |
| 2. DMPR `VariationalForm` | `LaplaceForm`'s sibling, `variational_form`, sharing its closed forms, point masses and discrete products through `project_form`; `nomode` and a `backend`. 717 tests. |
| 3. RxInfer `CVI()` | A `FallbackScheme`: `ExpectationRuleFallback` with `(q = VariationalForm, μ = VariationalForm(nomode = :leave))`. `fallback_form(scheme, backend)` hands the form the inference's backend, as `apply_fallback` now chooses the backend first. The form glue serves both forms (`form_laplace.jl`). The report reads "CVI, converged in n iterations, ELBO = …" and warns `:variational_not_converged`. A CVI section in the docs. |
| 4. EFP `ExpectationStrategy(backend)` | Deterministic: the score-form natural gradient over a point set from any BayesBase backend, the cost the negative ELBO (D53). Its tests use a quantile-midpoint backend, since EA is private. RxInfer's `ProjectedTo` form reports the strategy's ELBO and an outcome, `(method = :projection, family, logZ)`. |
| 5. Notebooks | Logistic Regression, "CVI: the best Gaussian": E[β] 10.29 against the grid's 10.35 (Laplace 7.2), free energy 5.44 against −log p(y) = 5.30 (Laplace 6.41), sd(β) 3.66 against 4.87. Constrained Variables, "A Beta posterior": E[p] 0.222 against the grid's 0.226 (logit-normal 0.298). |
| Tests | EA 1046, DeltaMessagePassingRules 717, RxInfer 15253 (28 new), ExponentialFamilyProjection 2808 (11 new; three slow performance items time out under load and pass alone), RxInfer's docs build; the five notebooks build. |
| Acceptance | `prototype/protocol/acceptance/cvi.jl`: CVI equals `gaussian_variational` of the bioassay's joint; −log p(y) < F(CVI) < F(Laplace); the zero-inflated Poisson's F(CVI) < F(Laplace) under mean field (24.80 < 24.86); a Beta-Bernoulli posterior projected onto the exact Beta. |

What it found:
- **`laplace` reported no convergence when started at its mode.** A Newton step of zero
  cannot pass the line search's sufficient-increase test. Fixed in EA: a step below the
  tolerance ends the iterations.
- **The natural-gradient iteration oscillates**: see D52.
- **EFP's sampled default is far off where quadrature is not.** On a Beta-Bernoulli target with
  conjugate answer Beta(9, 16), the default `ControlVariateStrategy` gives Beta(8.0, 14.3); a crude
  midpoint quadrature gives Beta(8.98, 15.97), Gauss–Jacobi the exact one.
- **KL(q‖p) underestimates the spread**, as expected: the bioassay's sd(β) is 3.66 under CVI
  against 4.87 exact.
- **universal-dev now takes ClosedFormExpectations from the registry** (0.4.3): EFP needs 0.4.2,
  and the BIASlab checkout is a stale 0.4.1 branch.

#### U7c. Exact refinements: moved to the P track

ClosedFormExpectations is BIASlab's (Mykola), so its adapter belongs to its 1.0 (P). The brief
found what it needs:
- **Support at the type level.** CFE fits BayesBase's type-level protocol: each closed form
  dispatches on a concrete integrand type. A support table must be written by hand, because the
  generic methods on `ExponentialFamilyDistribution` would claim everything.
- **The fallback never sends a recognisable integrand.** It asks for anonymous closures under a
  `FactorizedJoint`, and product routing (`prepare_product`, `F = Any` per component) never
  reaches a closed-form backend. Exact refinements need integrands such as `Fix1(logpdf, m_out) ∘
  exp` under a plain distribution.
- **Missing closed forms:** E[exp(x)] under a Gaussian, anything under a Beta, `log` under a
  LogNormal, and `GammaShapeRate` without conversion.
- **The local checkout is stale:** branch `fix/check-args-distributions`, 0.4.1, 15 commits
  behind `origin/main` (0.4.3); ExponentialFamilyProjection needs 0.4.2.

### U8. The inference report

Repositories: RxInfer.jl, with ReactiveMP and DeltaMessagePassingRules where an event or a
record is missing.

U5 lays its foundation (D46); this makes it a product:

1. **Presentation**: a summary that says in a few lines what was exact and what was approximated,
   and how; per node instance (its label in the model), not only per type.
2. **Diagnostics**: the parts most likely to be inaccurate (a Laplace iteration that did not
   converge, a stochastic backend, a fallback between two random variables under BP, a quantile
   rule as the last resort).
3. **Missing records**: joint Laplace convergence, deterministic nodes' free-energy terms,
   per-posterior provenance (perhaps through annotations).
4. **Other sections** of `InferenceReport`, such as timings beside the benchmark callbacks.
5. **The Universal Inference notebooks show the report.** Each of the four notebooks
   (RxInferExamples, see U6) gains a "What was approximated" part with `report = true`. Its
   hidden checks move from counting callbacks to the report, such as "the fallback never ran
   for the conjugate Poisson", and a notebook dedicated to the report joins them (U6 item 4).
   U8 is not done until they do.

#### What U8 delivered and found

Local on the `universal-inference` branches, one tested commit per step, not pushed.

| Item | Result |
|---|---|
| 1. Presentation | By posterior (the user's choice of layout). Each approximated posterior is a row group: how it was projected (Laplace, converged in n iterations, log Z), then the node instances whose messages formed it, with exact and fallback counts and the expectations. Instances are named as the model writes them, `y[3] ~ Bernoulli(p[3])`, `p[3] := logistic(·)`, `· := dot([1.0, -0.86], θ)`, built from ReactiveMP's interfaces and the GraphPPL labels of their variables. They are folded by shape, `y[1:300] ~ LogPrecisionMixture(z[1:300], m, s)`, and so are variables formed alike (the stochastic volatility path is three groups: `x[1]`, `x[2:29]`, `x[30]`). Records are kept per instance and per node type (`RxInfer.node_types`). |
| Tables, plain or pretty | `report_tables(section)`: `ReportTable`s of strings, with row groups and a level per row. `RxInfer.render_report_table` prints aligned text; RxInfer's PrettyTables extension overrides it through the `summarize_invokes` pattern, with rounded borders, row groups, warnings in red and notes in yellow, and adds `pretty_table(io, report)`. The same `result.report` prints as tables wherever PrettyTables is loaded. |
| 2. Diagnostics | `RxInfer.diagnostics(report)` returns `ReportDiagnostic(kind, level, subject, text)`s, folded by name. Warnings: `:laplace_not_converged`, `:joint_not_converged`, `:expectation_over_messages` (a fallback under BP averaging over another random variable's message). Notes: `:unprojected_messages` (D49), `:stochastic_expectation`, `:last_resort_expectation`. The last one uses a negative registry priority, read through BayesBase's new `expectation_backend_priority`, because a frozen `ExpectationFirstOf` keeps the order and not the priorities. Four of the six are provoked in RxInfer's tests; the joint's non-convergence and the last resort are not, as no small model reaches them (the joint's iterations are not configurable, and the last resort needs a type no other backend takes). |
| 3. Missing records | MessagePassingRulesBase `with_outcome` and `unwrap_outcome`: a fallback says how it computed an answer. ReactiveMP's three After rule events carry the `outcome`, and the old constructors remain. DeltaMessagePassingRules' joint Laplace returns `(method = :joint_laplace, converged, iterations)`, and ExponentialFamily Gaussians, as D50 does for marginals. Deterministic nodes' energy terms are covered by the joint inside their span. Provenance: each message event's target interface leads to its variable, `VariableRecord.incoming`. |
| 4. Timings | `TimingsReport`, `report = (:timings,)`, both by default: model creation, inference and per-iteration times. The time of exact rules, the fallback, projections and free-energy terms is measured from Before/After span pairs on a stack, so it is exclusive and the kinds add up; it is also broken down by node type and by variable. |
| 5. Notebooks | The four notebooks gain "What was approximated", with `result.report.approximations` shown and the hidden checks on the report: no warnings, the conjugate parts exact, Laplace converged, the deterministic nodes' rules and fallbacks. The zero-inflated Poisson notebook's callback counter is gone: its conjugate model's report is shown instead. A fifth notebook, "Reading the Inference Report", runs one model under mean field (no warnings) and under BP (`:expectation_over_messages`), and covers the report as data and the timings. RxInferExamples gains the `meta.jl` option `wide_output = true`, because Weave wraps output at 75 characters, which breaks tables. |
| Docs | RxInfer: a manual page, "What was approximated: the inference report", linked from the `report` keyword, the Laplace section and the Rule Not Found sharp bits. |
| Tests | MessagePassingRulesBase 956, ReactiveMP 15684, DeltaMessagePassingRules 697, BayesBase 6625, RxInfer 15225 (41 new); RxInfer's docs build with the new page; the five Universal Inference notebooks build in 3 min 14 s (`make examples-env … FILTER=laplace`). All run in an environment with every checkout by path and Distributions pinned at 0.25.132. |

What it found:
- **A stochastic backend cannot join a product expectation.** `MonteCarlo` in a `FactorizedJoint`, even with a `PointMass`, fails, because the product rule needs point sets for every component. A fallback with two inputs therefore cannot use it (EA, P).
- **Exclusive timings are honest, and they are mostly the engine's.** On the report notebook's model, a second run: the engine's products, streams and entropies take about 86%, the fallback and projections about 6% each.
- **A first run's timings include compilation.** The notebook times a second run.
- **Anonymous variables share one key**, `(anonymous_var_graphppl, nothing)`, so the report counts them as one posterior, shown as `· (anonymous)`.

### P. Ecosystem migration (parallel track)

- **ClosedFormExpectations 1.0** (BIASlab, Mykola): `CFEBackend`, `supports_expectation`, the
  `ExpectationExact` label; `mean(ClosedFormExpectation(), f, q)` deprecated for one cycle;
  Williams-product gradients once the gradient operation is settled. Depends on BayesBase only. Its
  own `Logpdf` (callable only for a `Distribution`) is replaced by `Base.Fix1(logpdf, p)`, and its
  recursive calls pass `backend` on instead of constructing `ClosedFormExpectation()`. It now
  also carries U7c, the exact refinements (see U7c for what the brief found).
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
- **ExponentialFamilyProjection**: a strategy holding any BayesBase backend landed in U7b
  (`ExpectationStrategy`, ReactiveBayes/ExponentialFamilyProjection.jl#120), for quadrature; its
  ClosedFormExpectations extension can become one once CFE's adapter exists.

### After the merge

What the effort set aside on purpose, to revise once the branches are merged and released:
- **The range of a Delta node's output.** The message from `p := logistic(…)` is a Gaussian on
  the real line, so a Bernoulli after it has no free energy (U6, finding 2).
- **Indexing a vector variable**, `θ[1]`, a large change in GraphPPL (U6, finding 4).
- **A `Binomial` factor node** (U6, finding 4).
- **`MonteCarlo` in product expectations.** A stochastic backend cannot take part in a
  `FactorizedJoint`, whose product rule needs point sets (U8).
- **Anonymous variables share one report key**, `(anonymous_var_graphppl, nothing)` (U8).

### Release order

Before any tag:
- **Releases in critical-path order:** BayesBase, MessagePassingRulesBase, EA, ReactiveMP,
  DeltaMessagePassingRules, ExponentialFamilyProjection, RxInfer, then RxInferExamples. Each
  release drops its cross-branch `[sources]` entries (`url`, `rev = "universal-inference"`).
- **EA becomes public and registered.** DeltaMessagePassingRules takes it by SSH URL while it is
  private.
- **The Distributions 0.25.133 break** of ExponentialFamily 2.9.0 (`distrname`) is fixed
  upstream; the user handles it separately.

### Critical path

```
U0 ─► U1 ─► U2 ─────────► U4 ─► U5 ─► U6 ─► U8 ─► U7b ─► (U7a on hold)
        └─► U3 (parallel) ─┘
        └─► P  (parallel; CFE 1.0 with U7c after U1, MPRA 1.1 after U2)
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
  question on their representation. The `LogNormal` and `LogitNormal` results are superseded by
  D41.
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

Added in U4:

- **D37. The universal rules live in DeltaMessagePassingRules**: `ExpectationRuleFallback` and
  `LaplaceForm`, plain types; the form constraint's ReactiveMP methods are RxInfer's
  (`src/constraints/form/form_laplace.jl`), as `ProjectionExt`'s are for `ProjectedTo`, and no rule
  package depends on the engine. Delta gains EA, its approximations' home after P.
- **D38. Automatic initialisation is U5's**, set by the one switch: it needs an RxInfer hook.
- **D39. A fallback declares its message's log scale** by returning `with_logscale(message,
  logscale)`, which an engine takes apart with MessagePassingRulesBase's `unwrap_fallback`;
  `ExpectationRuleFallback` declares one only where its message is exact, pending the normaliser
  (`:unnormalised_product`) that `LaplaceForm` reports.
- **D40. Fallback messages are existing types**: BayesBase's `ContinuousUnivariateLogPdf` where the
  support is known (towards `out` of a distribution node), MessagePassingRulesBase's
  `NodeFunctionLogPdf` elsewhere; towards a parameter not even the variate form is known.
- **D41. Every domain but the real line pushes forward to a `TransformedGaussian`**, which has
  `mean`, `var`, `entropy` and `mean(f, q)`; `LogNormal` and `LogitNormal` lack `mean(log, q)` in the
  ecosystem. Supersedes D27's `LogNormal` and `LogitNormal` as results.
- **D42. A joint without a rule is the joint Laplace approximation** of the node's log-density
  times its members' messages, in the product of their domains (`EA.Componentwise`), in
  expectation over the other clusters; its members are numbers for now.
- **D43. The switch is `infer(...; fallback = scheme)`**, a `FallbackScheme` such as
  `LaplaceApproximation()`: a keyword, not an option, and an object, not a boolean, so that U7's
  schemes are other values of it. A bare `Laplace` is Distributions' distribution.
- **D44. A fallback scheme's default backend is the session's** (revises D4 for this one use): the
  backends registered with BayesBase, frozen once per call of `infer` into a type-stable
  `ExpectationFirstOf` before the options reach the model, the dynamic choice paid once behind
  that barrier. It applies only under a scheme the user asked for, is recorded in the report, and
  is pinned by `LaplaceApproximation(expectation = …)`. EA registers `EADefault`'s rules when it
  loads, not its stochastic backends.
- **D45. Initialisation is a strategy any package can provide**: `RxInfer.InitializationStrategy`
  and `RxInfer.initial_marginal`, given as `initialization`; a specification applies first and is
  never overwritten. `FromPriors` sets marginals only.
- **D46. The report is collected from events and is composite**: `InferenceReport`, a section per
  subject, `approximations` first; off by default and free when off. Annotations cannot see which
  rule ran and have no hooks for joints, energies or forms. Its product is U8.
- **D47. Explicit settings win over a scheme**: `options.rulefallback`,
  `options.default_form_constraint`, `options.context.expectation` and an initialization strategy
  replace the scheme's, as a variable's own constraint replaces the default form.
- **D48. Every deterministic node goes through the one rule fallback**: Delta nodes, `dot`, `+`,
  `-` and `*` alike, its function coming from the engine. ReactiveMP's `getnodefn` gives a node
  whose form is a function that function, its inputs in interface order; a declared node's exact
  rules keep precedence.
- **D49. A projection that finds no mode is not used for a message**: `LaplaceForm` leaves a
  message unprojected when Newton does not converge, so an exact rule downstream does not match
  and the fallback pulls back through the node; a marginal keeps the result, its record saying
  `converged = false` for the report.
- **D50. The Laplace projection gives ExponentialFamily's Gaussians on the real line**:
  `LaplaceForm` turns ExpectationApproximations' `Normal` and `MvNormal` into `NormalMeanVariance`
  and `MvNormalMeanCovariance`, the types the Gaussian nodes' rules take. EA itself stays on
  Distributions, ExponentialFamily being its weak dependency; a Gaussian on another domain stays a
  `TransformedGaussian`.
- **D51. CVI is a projection, with no cavity**: `VariationalForm`, a form constraint in
  `LaplaceForm`'s place, minimises KL(q‖p̃) of a variable's product of messages over Gaussians in
  the variable's unconstrained coordinates. The rules stay as they are.
- **D52. The variational Gaussian maximises its bound by Newton's method**, not by the
  natural-gradient iteration. The bound, `Σ wᵢ log p̃(m + L zᵢ) + log det L`, is a quadrature of a
  standard Gaussian at fixed points, a smooth function of the mean and the log-Cholesky factor,
  and `laplace` maximises it. At the maximum E_q[∇ log p̃] = 0 and E_q[∇² log p̃] = -Σ⁻¹, the CVI
  fixed point. *(Revised in U7b: the natural-gradient iteration in mean parameters fell into a
  two-cycle at full step and crawled for hundreds of iterations on a 2-D logistic regression,
  where Newton takes six.)* `expectation_gradient` stays an open question for closed-form
  gradients (P).
- **D53. ExponentialFamilyProjection's deterministic strategy holds a BayesBase backend**,
  `ExpectationStrategy(backend)`, and depends on the protocol only, not on EA (D2).

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
- ~~**The universal rules package**~~: DeltaMessagePassingRules, with the form constraint's
  methods in RxInfer (D37).
- ~~**The fallback signature**~~: positional, through `call_fallback` (D31).
- ~~**Where the Laplace projection lives**~~: a form constraint reporting its normaliser through
  ReactiveMP's hook, composed by RxInfer's option (D33); the constraint itself is U4's.
- **Domains**: how a variable learns its support when several nodes connect to it, or when
  no neighbour carries a support (U0.3b found the prior factor is enough when there is one).
- ~~**The user-facing switch**~~: `infer(...; fallback = LaplaceApproximation())` (D43).
- ~~**Default on or off**~~: off; `fallback = nothing` by default (the user's call).
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
| DeltaMessagePassingRules.jl | `~/Projects/Julia/DeltaMessagePassingRules.jl` (local branch `universal-inference`) |
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
- **Pitfall: `Pkg.test` re-resolves.** Its sandbox reads a path dependency's own `[sources]`, so
  RxInfer's tests took EA through DeltaMessagePassingRules' SSH URL, at a stale cached clone without
  `register_backends!`, while `Pkg.status` showed EA by path. Since then (November 2026),
  Distributions 0.25.133 drops `distrname`, which ExponentialFamily 2.9.0 imports, so a fresh
  resolve no longer precompiles. To test against the local checkouts, use an environment that
  develops all of them by path, with Distributions pinned at 0.25.132, and RxInfer's test targets
  added. Run `runtests.jl` from the package's `test/` with `--project` set to that environment.

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
| U4 | DeltaMessagePassingRules, EA, MessagePassingRulesBase, ReactiveMP, RxInfer | drafts on the pushed `universal-inference` branches; DeltaMessagePassingRules takes EA by its SSH URL while EA is private (see "What U4 delivered and found") | [DeltaMessagePassingRules.jl#2](https://github.com/ReactiveBayes/DeltaMessagePassingRules.jl/pull/2), and the U3 pull requests updated |
| U5 | RxInfer, EA, DeltaMessagePassingRules | done on the pushed `universal-inference` branches (see "What U5 delivered and found") | the U3 and U4 pull requests updated |
| U6 | RxInferExamples | partly done: five Universal Inference notebooks on the switch, with the report and hidden checks (see "The notebooks on the switch"); in progress: a pendulum, a hierarchical model and a Poisson count series, each under Laplace and CVI, and a better story for "Simple Nonlinear Node" | [RxInferExamples.jl#109](https://github.com/ReactiveBayes/RxInferExamples.jl/pull/109) |
| U7 | DeltaMessagePassingRules, EA, ExponentialFamilyProjection, RxInfer, RxInferExamples | U7a (cavity methods) on hold; U7b (CVI, deterministic EFP) done on the pushed `universal-inference` branches (see "What U7b delivered and found"); U7c moved to P | [ExponentialFamilyProjection.jl#120](https://github.com/ReactiveBayes/ExponentialFamilyProjection.jl/pull/120), and the open pull requests updated |
| U8 | RxInfer, ReactiveMP, MessagePassingRulesBase, DeltaMessagePassingRules, BayesBase, RxInferExamples | done on the pushed `universal-inference` branches (see "What U8 delivered and found") | the open pull requests updated |
| P | CFE (with U7c), MPRA, rule packages | not started | |

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
