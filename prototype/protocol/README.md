# Prototype of the expectation protocol (U0.2)

A prototype of the protocol planned for BayesBase (PLAN.md, milestone U0.2), in two small
packages so that the code and its tests move into BayesBase (U1) and ExpectationApproximations
(U2) without rewriting. Deleted once those land.

- `ProtoBase/` stands in for the BayesBase additions: `expectation`, `prepare_expectation`,
  `supports`, labels, `withrng`, `Logpdf`, point sets, products over `FactorizedJoint`,
  `FirstOf` and `DynamicRouter`. Nothing is exported.
- `ProtoBackends/` stands in for ExpectationApproximations and ClosedFormExpectations:
  `ClosedForms`, `GaussHermite`, `QuantileLegendre`, `MonteCarlo`, `default_router`.
- `acceptance/` runs two notebooks of the Universal Inference section with their fallback
  rewritten with the protocol, and compares the posteriors with the originals.
- `experiments/` measures the cost of `DynamicRouter` and tries the operation axis.
- `RESULTS.md` has the measurements and findings.

Run everything in the shared environment `~/Projects/Julia/universal-dev` (Julia 1.11+):

```sh
julia --project=~/Projects/Julia/universal-dev ProtoBase/test/runtests.jl
julia --project=~/Projects/Julia/universal-dev ProtoBackends/test/runtests.jl
julia --project=~/Projects/Julia/universal-dev acceptance/compare.jl
julia --project=~/Projects/Julia/universal-dev experiments/dynamic_router.jl
julia --project=~/Projects/Julia/universal-dev experiments/operation_axis.jl
```

`ProtoBase` also runs on Julia 1.10, in a temporary environment:

```sh
julia +1.10 -e 'using Pkg; Pkg.activate(temp = true); Pkg.develop(path = "ProtoBase"); Pkg.test("ProtoBase")'
```
