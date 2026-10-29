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

1. Check out every repository involved locally (BayesBase.jl is not checked out yet) and set up
   the shared development environment ([Working across repositories](#working-across-repositories)).
2. Prototype the BayesBase protocol in a scratch environment: `expectation`, `supports`,
   labels, `withrng`, `FirstOf`, `DynamicRouter`, with a toy closed-form backend and a
   Gauss-Hermite stub. Confirm with `@inferred`, JET and `@allocated` that `FirstOf`'s choice
   and `haslabel` fold at compile time and that `withrng` does not allocate. Measure the cost of
   `DynamicRouter` per call.
3. **Laplace spike in today's RxInfer**, before any refactor: `rulefallback =
   NodeFunctionRuleFallback()` plus a hand-written Laplace form constraint on every variable,
   on logistic regression and a hierarchical model with a Gamma precision. Goal: find every
   place the engine gets in the way (missing context in fallbacks, marginal rules, free energy,
   initialisation, constrained domains) with real models, and record them below.

Done when: the prototype's measurements and the spike's findings are recorded in
[Decisions](#decisions) and the milestones below are adjusted.

### U1. BayesBase: the expectation protocol

Repository: BayesBase.jl. Additive, minor release.

1. `expectation(backend, f, q)`; `supports` at type level (hook) and instance level (API).
2. Labels: `ExpectationLabel`, `Exact`, `Stochastic`; `haslabel` per backend and per pair;
   `isexact`, `isstochastic`.
3. `withrng`, `getrng`; the integrand `Logpdf`.
4. Routers: `FirstOf` with `prefer` and `exclude`; opt-in `DynamicRouter` with
   `register_backend!` and `which_backend`.
5. Decide the operation axis (`expectation`, `expectation_gradient`, `mean_cov`) before
   release; adding it later is breaking.

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
4. **A default form constraint for all random variables**, composed where
   `EnsureSupportedFunctionalForm` is composed today, so "every variable Gaussian (in its
   unconstrained space)" can be one setting instead of one line per variable.
5. **Damping** as a stream postprocessor, needed by EP and loopy schedules later.

Done when: existing behaviour unchanged (`NodeFunctionRuleFallback` tests pass as they are),
new hooks tested, documentation updated (fallbacks, context, form constraints).

### U4. Universal Laplace rules

Repository: a new rules package (name open, see [Open questions](#open-questions)), depending
on MessagePassingRulesBase, BayesBase and EA.

1. **Stochastic nodes**, any node with a `nodefunction`: the message towards a target is the
   expected log-density `log m(x) = E_q(y)[log f(x, y)]` over the other inputs, computed through
   `ctx.expectation` (a plug-in value when an input is a point mass, an exact sum when it is
   categorical).
2. **Laplace projection of marginals** in unconstrained coordinates: mode and Hessian of the
   sum of incoming log-densities plus the log-Jacobian; the domain from the prior's support.
3. **Deterministic nodes**: a default Delta method when the user gives none, `Linearization()`
   for the Laplace scheme.
4. **Average energies** for the free energy on the universal path.
5. **Automatic initialisation** from the priors (Gaussian in unconstrained space).

Done when, with no rules, methods or initialisation written by the user:
- on linear-Gaussian models the posterior means equal the Kalman smoother's (mean-field
  Laplace gives exact means and smaller variances; exact variances come with U7);
- the canonical models of [VISION.md](VISION.md#what-success-looks-like) run, and their
  posterior moments agree with long NUTS runs within tolerances recorded in the tests;
- conjugate models never trigger a fallback (checked through rule annotations).

### U5. RxInfer integration

Repository: RxInfer.jl.

1. **One user-facing switch** for the universal scheme (keyword or option, name open) that
   sets the fallback, the default form constraint, the expectation backend and initialisation.
2. **The context**: `ctx.expectation = withrng(FirstOf(CFEBackend(), EADefault()), rng)` by
   default, overridable from the inference options and per node.
3. **An approximation report** on the inference result: which nodes ran exact rules, which ran
   fallbacks, with which projection and which backend labels (via annotations or callbacks).
4. **Error hints**: `RuleNotFoundError` and the Delta-node hint point to the universal switch.
5. **The default**: decide whether the universal scheme is on by default, after U6.

Done when: the canonical models run from `infer` with one switch; the report is documented;
existing tests unchanged.

### U6. RxInferExamples: universal inference

Repository: RxInferExamples.jl.

1. A new "Universal inference" section, starting with a purely Laplace example: logistic
   regression, a Poisson count time series, a pendulum state-space model, a hierarchical model
   with a Gamma precision; each compared with NUTS.
2. "Simple Nonlinear Node" rewritten without its hand-written rules.
3. Later: the same models under EP, and with exact refinements via ClosedFormExpectations.

Done when: the examples run in the examples' CI and show the approximation report.

### U7. Beyond Laplace

Repositories: the universal rules package, EA, ReactiveMP.

1. **Laplace propagation and EP**: belief-propagation structure with cavities (rules that read
   the message on their own edge, as Probit's EP rules already do), Gaussian division with
   positive-definite correction, damping. Same code, two projections: Laplace or moment
   matching through EA's quadrature. Acceptance: exact Kalman smoother (means and variances)
   on linear-Gaussian models.
2. **CVI and natural-gradient projection** through `expectation_gradient`; ExponentialFamilyProjection
   as the projection to families beyond the Gaussian.
3. **Exact refinements**: ClosedFormExpectations as the first backend in `FirstOf`, showcased on
   Delta nodes computing `exp` or `log` against a `Logpdf`.

### P. Ecosystem migration (parallel track)

- **ClosedFormExpectations 1.0** (BIASlab, Mykola): `CFEBackend`, `supports`, the `Exact`
  label; `mean(ClosedFormExpectation(), f, q)` deprecated for one cycle; Williams-product
  gradients once the gradient operation is settled. Depends on BayesBase only.
- **MessagePassingRulesApproximations 1.1**: a shim re-exporting EA's types with
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
  routing calls such as `expectation(::Logpdf{<:Laplace}, ::Normal)` therefore cannot live in a
  backend package.
- **D2. Thin protocol in BayesBase.** BayesBase owns `expectation`, `supports`, labels,
  `withrng`, `Logpdf` and the routers. No new dependencies. EA and ClosedFormExpectations
  implement it on their own backend types; there is no dependency between EA and CFE.
- **D3. Routing is a value.** `FirstOf(backends...; prefer, exclude)` is itself a backend;
  candidates and preferences live in its type, so the choice is made at compile time. Several
  policies coexist (RxInfer's, ExponentialFamilyProjection's, a user's). Routers nest.
- **D4. Automatic choice is opt-in.** `DynamicRouter` picks among loaded backends, ranked by
  labels, then priority, then name, never load order. Type unstable by design; never a
  default; never used inside library code; `FirstOf(DynamicRouter())` freezes it.
- **D5. Labels are traits, not a type hierarchy.** `Exact` (equals the expectation up to
  rounding) and `Stochastic` (depends on an RNG) in BayesBase, declared per backend or per
  pair, conservative (true only when guaranteed), forwarded by composite backends, tested by
  the conformance suite. Packages may define further labels.
- **D6. The RNG is a property of the backend.** Stochastic backends hold it; `withrng`
  fills it from above, recursively, only where unset. Calls never pass an RNG.
- **D7. Rules read `ctx.expectation` and nothing else.** The engine resolves backend, RNG
  and per-node overrides before rules run. A method that *is* a node's algorithm (the
  Delta node's `Unscented()`) stays in the algorithm.
- **D8. `Logpdf` is a struct**, not an alias of `Base.Fix1{typeof(logpdf)}`, whose
  constructor would itself be piracy.
- **D9. Names follow Base style** (`haslabel`, `isexact`, `withrng`), tentatively.
- **D10. Exact first, at every level.** Exact rules before fallbacks; exact backends before
  approximate ones in the default router.
- **D11. The universal path is a fallback.** It never replaces an existing rule; it is
  consulted only when resolution finds none.
- **D12. Laplace first.** The first universal scheme is mean-field Laplace (no message
  division, no damping needed); Laplace propagation and EP follow in U7.

The protocol, as decided (BayesBase):

```julia
function expectation end                                   # expectation(backend, f, q) = E_q[f]

supports(backend, ::Type, ::Type) = false                  # hook for backend owners
supports(backend, f, q) = supports(backend, typeof(f), typeof(q))

abstract type ExpectationLabel end
struct Exact <: ExpectationLabel end
struct Stochastic <: ExpectationLabel end
haslabel(backend, label) = false
haslabel(backend, ::Type{F}, ::Type{Q}, label) where {F, Q} = haslabel(backend, label)
haslabel(backend, f, q, label) = supports(backend, f, q) && haslabel(backend, typeof(f), typeof(q), label)
isexact(backend, f, q) = haslabel(backend, f, q, Exact())

withrng(backend, rng) = backend                            # filled from above, once
getrng(backend) = nothing

struct Logpdf{D}; dist::D; end

struct FirstOf{B <: Tuple, P <: Tuple, X <: Tuple}         # a router is a backend
    backends::B; prefer::P; exclude::X
end
```

## Open questions

- **Operation axis** in `supports` and `haslabel` (`expectation`, `expectation_gradient`,
  `mean_cov`): now or never, since adding it later is breaking.
- **Gradients**: `expectation_gradient(backend, f, q, space)`; the parameter-space types live
  in ExponentialFamily, so BayesBase can only declare it with an untyped `space`.
- **`Differentiable` as a core label**, needed by CVI and natural-gradient schemes.
- **Exports** of BayesBase: which names are exported, which stay qualified.
- **Prepared path**: `prepare_expectation` in EA only, or reserved in BayesBase.
- **The universal rules package**: a new package (`UniversalMessagePassingRules.jl`?), part
  of EA as an extension on MessagePassingRulesBase, or part of DeltaMessagePassingRules.
- **The fallback signature** in MessagePassingRulesBase: extra positional arguments, a
  call object, or the context inside `RuleArgs`.
- **Where the Laplace projection lives**: a form constraint in RxInfer (as `ProjectedTo`
  is), or a ReactiveMP-level projection hook.
- **Domains**: how a variable learns its support when several nodes connect to it.
- **The user-facing switch**: a keyword of `infer`, an option, or a constraint-level wildcard.
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
| BayesBase.jl | not checked out yet: clone to `~/Projects/Julia/BayesBase.jl` |
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

Tests (ReTestItems or TestItemRunner), Aqua with ambiguity and piracy checks, JET on Julia
1.12+, allocation tests on hot paths, documentation, and a status entry here.

## Status

| ID | Repository | State | Pull request |
|---|---|---|---|
| U0 | EA (hub), scratch | started: plan written | |
| U1 | BayesBase | not started | |
| U2 | EA | not started | |
| U3 | MessagePassingRulesBase, ReactiveMP | not started | |
| U4 | universal rules (new) | not started | |
| U5 | RxInfer | not started | |
| U6 | RxInferExamples | not started | |
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
