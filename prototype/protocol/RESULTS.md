# Results of the protocol prototype (U0.2)

Measured on 2026-11-02 with Julia 1.13.3 (and 1.10.12 for `ProtoBase`), in the
`~/Projects/Julia/universal-dev` environment: BayesBase 1.6.0, RxInfer 6.1.1, ReactiveMP 7.0.1,
MessagePassingRulesBase 1.1.3. Every claim below is a test or a script in this folder.

## Summary

The design in PLAN.md holds. Routing and labels are decided by inference, prepared
expectations do not allocate and pass dual numbers, and the notebooks' hand-written fallback
rewritten with the protocol gives the same posteriors. The prototype also settled two open
questions and found three things the design had not said yet.

| # | Measurement | Result | Where |
|---|---|---|---|
| 1 | Routing folds | `which_backend(FirstOf(...), F, Q)` compiles to `return <backend>`; `@inferred` and JET `@test_opt` clean | `ProtoBase/test/firstof_tests.jl` |
| 2 | Labels fold | `isexact` compiles to `return true` / `return false`, for plain backends, routers and products | `ProtoBase/test/labels_tests.jl`, `products_tests.jl` |
| 3 | `withrng` | 0 bytes, nested routers included; fills only unset RNGs | `ProtoBase/test/firstof_tests.jl`, `ProtoBackends/test/routing_tests.jl` |
| 4 | Prepared path | 0 bytes for point sets, products of Categorical × MvNormal × PointMass, and the mixture notebook's integrand differentiated twice with ForwardDiff | `pointset_tests.jl`, `products_tests.jl`, `routing_tests.jl` |
| 5 | `DynamicRouter` cost | 1.4–1.8 µs and 1.4–2.3 KB per call against 8.7 ns and 0 bytes for `FirstOf`; freezing it into a `FirstOf` recovers the static numbers | `experiments/dynamic_router.jl` |
| 6 | Collisions | with nothing exported, `using RxInfer, ProtoBase, ProtoBackends` keeps `Stochastic` and `isstochastic` as RxInfer's | `ProtoBackends/test/collisions_tests.jl` |
| 7 | Operation axis | can be added later without breaking anything: 11 of 11 checks | `experiments/operation_axis.jl` |
| 8 | Acceptance | mixture and zero-inflated Poisson notebooks: posteriors equal to ≤ 1.4e-15 (required 1e-10); compiled inference 6.04 → 6.18 s and 1.31 → 1.17 s | `acceptance/compare.jl` |

Tests: `ProtoBase` 247 of 247 on Julia 1.13 and 225 of 225 on 1.10 (the public-name checks
differ before 1.11); `ProtoBackends` 129 of 129 on 1.13. Aqua runs with ambiguity and piracy
checks on: none found.

## Settled

- **Exports (D18).** The protocol names are public, declared with `Compat.@compat public`, and
  never exported. This removes every collision found by the briefing: `Stochastic` and
  `isstochastic` exported by MessagePassingRulesBase and ReactiveMP, `expectation` exported by
  MessagePassingRulesApproximations PR #2 and re-exported by RxInfer.
- **Operation axis (D21).** PLAN said "now or never". The experiment adds a four-argument
  `supports(backend, op, F, Q)` after the fact: for `expectation` it forwards to the released
  three-argument form, so existing backends and callers are untouched, and new operations are
  opt-in. Routing per operation still folds. Decision: do not add it now.

## Found

- **A preparation is chosen without knowing the integrand.** Routing depends on the type of
  `f`, but `prepare_expectation(router, q)` has no `f`. Convention: `supports(backend, Any, Q)`
  means "supports every integrand", which is what a numerical backend declares with
  `supports(::B, ::Type, ::Type{<:Family})`. A prepared `FirstOf` prepares the backend chosen
  for `F = Any`, and still sends an integrand with a closed form to it, decided by inference.
- **Point sets are the common currency of numerical backends.** Product rules over independent
  components need every component's preparation to be a weighted point set. `PointSet` and
  `ProductPointSet` belong in the protocol, with a generic, dual-safe sum that starts from its
  first term. A preparation that is not a point set (a closed form, Laplace) cannot be a
  component of a product, and says so.
- **Products are routed component by component.** The first iteration of the zero-inflated
  Poisson notebook integrates over `(PointMass, Gamma)`: no single backend covers both, a
  router covering each component does. Labels of a product combine per label:
  `combine_label(Exact(), a, b) = a & b`, `combine_label(Stochastic(), a, b) = a | b`, and
  conservatively for labels defined elsewhere.
- **Default `expectation` goes through the preparation**, with a guard: a backend with neither
  a closed form nor a preparation throws `NoBackendError` instead of a stack overflow.
  Backends define `prepare_expectation` and `expectation` for specific families, never for all
  `q`, or the generic product methods become ambiguous; routers define the product methods
  explicitly.
- **Quantile quadrature converges algebraically** for Beta and Gamma priors (quantile functions
  like √u and −log(1 − u) at the ends): errors 6.9e-5, 2.7e-6, 9.8e-8 for a Beta(2, 5) mean
  at 16, 64, 256 points, and 5.0e-3, 3.3e-4, 2.0e-5 for a Gamma(2, 2) mean. Fine as a
  universal fallback, but Gauss–Laguerre and Gauss–Jacobi rules (PLAN U2) are worth having.
- **Measuring allocations on Julia 1.10.** `@allocated` in a generic helper counts the boxed
  result on 1.10, so the tests measure with BenchmarkTools' `@ballocated`, checked against an
  allocating control. On 1.10, `map` with a closure over nested routers allocated in `withrng`;
  an explicit tuple recursion does not.

## Caveats

- The `DynamicRouter` timings for `sin` under a Gamma are not like-for-like: in the five-backend
  registry, priorities make it pick the 8-point rule.
- Whole-script times in the acceptance run include package loading, which the first script
  pays for; only the compiled inference times are comparable.
