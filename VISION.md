# Vision: every model runs

> RxInfer returns an answer for every model that can be written down. Exact where the
> mathematics allows it, approximate everywhere else, and always clear about which is which.

This document states where we want the ReactiveBayes ecosystem to be and why. [PLAN.md](PLAN.md)
says how we get there, in which repositories and in which order. Both live on the
`universal-inference` branch of ExpectationApproximations.jl, the hub of this effort.

## Where we are

RxInfer is fast and exact on conjugate models, and it fails loudly on most others. Today:

- A non-conjugate pair of nodes ends in `RuleNotFoundError`. The user has to find, choose and
  configure an approximation, or write rules by hand: 14 of the examples in RxInferExamples.jl
  ship hand-written rules.
- A nonlinear function, `y := f(x)`, has no rule at all until the user picks `Linearization()`,
  `Unscented()` or `CVIProjection()` for that node.
- "Treat every variable as Gaussian" cannot be said. Form constraints are per variable, with
  no wildcard; `q(x)::ProjectedTo(NormalMeanVariance)` must be written for each one.
- ReactiveMP v7 dropped Laplace, CVI, importance sampling and several quadratures. The rule
  fallback that remains, `NodeFunctionRuleFallback`, collapses every other input to its mean.
- Loopy and variational models need hand-written initialisation.

Each step is reasonable on its own. Together they mean that the first contact of a new user
with a slightly unusual model is an error message, not a posterior.

## Where we want to be

```julia
result = infer(model = my_model(), data = (y = y,))     # any model: it runs
```

- **Every model runs.** Any model GraphPPL can express gets approximate posteriors from `infer`
  without hand-written rules, methods or initialisation.
- **Exact first.** Wherever an exact or conjugate rule exists it is used, unchanged. The universal
  machinery only fills the gaps, so nothing that works today gets worse or slower.
- **Transparent.** The result says which nodes ran exact rules, which ran approximations, and
  which approximations; diagnostics flag the parts most likely to be inaccurate.
- **Refinable.** An advanced user improves any part: a better method for one node, an exact
  closed form for one function (a Delta node computing `exp`, `log` or `sin` against a
  log-density), a different universal scheme for the whole model.
- **Extensible by others.** Researchers (Mykola's closed forms, new projection schemes) add
  methods as packages that plug in, without touching the engine or the rules.

## The idea

Message passing can be automated completely if every belief is Gaussian. A factor of any
kind is then only ever asked one question: given Gaussian beliefs about its neighbours, what
Gaussian should it send? Every factor that has a log-density, and every deterministic function,
can answer that numerically. The literature has three classic ways to answer it, and they share
almost all of their machinery.

### Three choices, one substrate

| Choice | Options |
|---|---|
| Message-passing structure | variational (mean-field VMP) · belief propagation / EP (cavity and division) |
| Local Gaussian projection | Laplace (mode and Hessian) · moment matching (EP) · natural gradient (CVI, NGD) |
| How expectations are computed | exact closed forms · deterministic quadrature · sampling |

- **Laplace.** Fit a Gaussian at the mode of the local log-density, with the negative inverse
  Hessian as covariance. Cheap and deterministic; needs derivatives; mode-seeking. Inside a
  belief-propagation schedule this is *Laplace propagation* (Eskin, Smola and Vishwanathan,
  2003): its fixed points are the local optima of the posterior, and on jointly Gaussian models
  it coincides with belief propagation, so it reproduces the Kalman smoother exactly. In a
  mean-field schedule it is *variational Laplace* (Friston et al.), and it is one of the two
  approximations Extended VMP used in ForneyLab (Akbayrak, Bocharov and de Vries, 2021).
- **Moment matching (EP).** Match the mean and covariance of the tilted distribution, the
  factor times the cavity. Mass-covering, needs no derivatives, handles non-smooth functions
  (ReLU, `abs`); needs integrals, which quadrature or cubature provide; can produce negative
  precisions and needs damping (Minka; Roche on numerical schemes for EP).
- **Natural gradient (CVI, the Bayesian learning rule).** Move the Gaussian along the natural
  gradient of the expected log-density (Khan and Rue, 2023). Works for any differentiable
  factor and any exponential family, not only Gaussians; the route ExponentialFamilyProjection
  already takes.

All three reduce to the same few numbers about a function `f` and a Gaussian `q`:

- moment matching needs `E_q[f]`, `E_q[f·x]` and `E_q[f·xxᵀ]` (Z, mean, covariance of `f·q`);
- variational schedules need `E_q[log f]`, and CVI its gradient;
- Laplace needs the mode and Hessian of `log f + log q`.

That substrate is the **expectation protocol** in BayesBase, implemented by
ExpectationApproximations (quadrature, cubature, sampling, Laplace) and ClosedFormExpectations
(exact closed forms). Because the schemes only ever call the protocol, every exact closed form
added to ClosedFormExpectations improves every scheme at once, without a single new rule.

### What "any model" means

"Every belief is Gaussian" is the core, made precise so it holds for real models: every
belief is **Gaussian or finite categorical, possibly in transformed coordinates, possibly over
a block of variables**.

- **Unconstrained continuous variables** are Gaussian. Strongly correlated parameters belong in
  one block (one vector variable), so the Gaussian keeps their correlation.
- **Constrained continuous variables** (precisions, rates, scales, probabilities, simplex
  weights) are Gaussian in an unconstrained space: log for positive, logit for the unit
  interval, a softmax-type map for the simplex. The domain is known from the support of the
  variable's prior. The reported posterior is the pushforward (a log-normal for a precision).
  The transform lives in the projection and the expectation backends, not in the model: the
  user writes `τ ~ Gamma(...)`, not its logarithm.
- **Finite discrete variables** stay categorical. Their expectations are exact sums, so a
  mixture model needs no approximation on its discrete part.
- **Deterministic functions** pull the message they receive back through the function:
  towards an input `x`, the message is `log m_out(f(x))`, which works whatever the downstream
  node is (a Beta message from a Bernoulli as well as a Gaussian), and is exact under belief
  propagation. Linearisation and the unscented transform remain available where every
  neighbouring message is Gaussian; they cannot serve `Bernoulli(logistic(x))`.
- **Out of scope at first:** discrete variables with unbounded support, and multimodal
  posteriors that a single Gaussian cannot represent. These keep needing dedicated methods.

## Principles

1. **Exact first, approximate always.** Specific rules win; the universal path is a fallback,
   never a replacement. "Runs" is the minimum, "exact" the preference.
2. **One substrate.** All schemes compute through the expectation protocol. No scheme carries
   its own copy of quadrature, Gauss-Hermite or sigma points.
3. **Configured from above.** Rules call `expectation(ctx.expectation, f, q)` and nothing else.
   The engine decides the backend, its RNG and per-node overrides. Rules never pick defaults.
4. **Deterministic by default.** The default universal scheme uses no random numbers; sampling
   is opt-in and labelled `ExpectationStochastic`.
5. **Honest results.** Labels (`ExpectationExact`, `ExpectationStochastic`) travel with the
   computation, so a result can report how it was obtained, and tests can choose their tolerances.
6. **No piracy, no global magic.** Every method is defined on a type its package owns.
   Results never depend on which packages happen to be loaded, unless the user asked for it.
7. **Fast where it matters.** Type-stable, allocation-free hot paths for scalars and static
   arrays. Universality must not tax the conjugate models that are fast today.

## What success looks like

- **Coverage.** A suite of canonical non-conjugate models runs with plain `infer`, no rules,
  no methods, no initialisation: logistic and Poisson regression, a nonlinear state-space model
  (pendulum), stochastic volatility, a hierarchical model with Gamma-distributed precisions,
  a Gaussian mixture, and the "Simple Nonlinear Node" example without its hand-written rules.
- **Correctness.** On linear-Gaussian models the universal path reproduces the exact Kalman
  smoother to rounding error. On any model it equals the same approximation computed another
  way (a Laplace fit of the hand-written posterior). On conjugate models no fallback is ever
  triggered.
- **Quality, measured honestly.** The gap of each approximation to long NUTS runs or exact
  grids is published per model and per method, and guarded against regressions. Laplace is
  mode-seeking and will miss skew; that is reported, not hidden, and is what EP and exact
  refinements are for.
- **Transparency.** Every result can list the nodes that used an approximation and which one.
- **Speed.** Conjugate models are not slower than today. The universal path has published
  benchmarks and allocation-free kernels for the common sizes.
- **A showcase.** RxInferExamples.jl has a "Universal inference" section, starting with a
  purely Laplace example, then EP, then exact refinements via ClosedFormExpectations.

## Who does what

| Package | Role in the vision |
|---|---|
| BayesBase | The expectation protocol: `expectation`, `supports_expectation`, labels, `with_expectation_rng`, `ExpectationFirstOf` |
| ExpectationApproximations | The numerics: quadrature, cubature, sampling, Laplace, local Gaussian projections, domain transforms; the conformance suite |
| ClosedFormExpectations | Exact closed forms as a backend; the accuracy oracle in tests |
| MessagePassingRulesBase | Rule and fallback interfaces: fallbacks that see the context, the node and the algorithm |
| ReactiveMP | The engine: consults fallbacks for messages, marginals and energies; projections; damping |
| DeltaMessagePassingRules | The universal fallback and projection (`ExpectationRuleFallback`, `LaplaceForm`): Laplace, then EP and CVI, for any stochastic or deterministic node |
| RxInfer | The user-facing switch, defaults, automatic initialisation, the approximation report |
| RxInferExamples | The showcase and the acceptance tests in the wild |
| ExponentialFamilyProjection | Projection beyond Gaussians; becomes one of the projection operators |
| MessagePassingRulesApproximations | Deprecated through a shim once ExpectationApproximations covers it |

## Non-goals

- Replacing exact rules. They stay the fast path.
- Universal *exact* inference. Some models only admit approximations; the goal is that they run.
- A general-purpose MCMC engine. Sampling backends exist for expectations, not as an inference
  scheme of their own.

## References

- Eskin, Smola, Vishwanathan. [Laplace Propagation](https://papers.nips.cc/paper/2444-laplace-propagation). NIPS 2003.
- Minka. [Divergence Measures and Message Passing](https://www.microsoft.com/en-us/research/publication/divergence-measures-and-message-passing/). MSR-TR-2005-173.
- Winn, Bishop. [Variational Message Passing](https://jmlr.org/papers/v6/winn05a.html). JMLR 2005.
- Knowles, Minka. [Non-conjugate Variational Message Passing](https://proceedings.neurips.cc/paper/2011/file/5c936263f3428a40227908d5a3847c0b-Paper.pdf). NIPS 2011.
- Akbayrak, Bocharov, de Vries. [Extended Variational Message Passing for Automated Approximate Bayesian Inference](https://www.mdpi.com/1099-4300/23/7/815). Entropy 2021.
- Khan, Rue. [The Bayesian Learning Rule](https://www.jmlr.org/beta/papers/v24/22-0291.html). JMLR 2023.
- Roche. [On numerical approximation schemes for expectation propagation](https://ar5iv.labs.arxiv.org/html/1611.04416).
- Friston et al. [DEM: a variational treatment of dynamic systems](https://www.fil.ion.ucl.ac.uk/spm/doc/papers/DEM_A_variational_treatment_of_dynamic_systems.pdf). NeuroImage 2008.
- Davison, Ortiz. [FutureMapping 2: Gaussian Belief Propagation for Spatial AI](https://arxiv.org/abs/1910.14139). 2019.
