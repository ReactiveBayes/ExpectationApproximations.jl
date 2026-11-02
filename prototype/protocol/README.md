# Prototype of the expectation protocol (U0.2)

The protocol was prototyped here for BayesBase (PLAN.md, milestone U0.2) and now lives in
BayesBase itself (U1, `src/expectations/` on its `universal-inference` branch, to be released
as 1.7). The prototype of the protocol, `ProtoBase`, is in the history of this branch.

- `ProtoBackends/` stands in for ExpectationApproximations and ClosedFormExpectations, on top of
  the BayesBase protocol: `ClosedForms`, `GaussHermite`, `QuantileLegendre`, `MonteCarlo`,
  `default_router`. Its code and tests move into EA in U2.
- `acceptance/` runs two notebooks of the Universal Inference section with their fallback
  rewritten with the protocol, and compares the posteriors with the originals.
- `experiments/` measures the cost of `DynamicRouter` and tries the operation axis.
- `RESULTS.md` has the measurements and findings of U0.2, made against `ProtoBase`.

Run everything in the shared environment `~/Projects/Julia/universal-dev` (Julia 1.11+), which
uses the local BayesBase checkout:

```sh
julia --project=~/Projects/Julia/universal-dev ProtoBackends/test/runtests.jl
julia --project=~/Projects/Julia/universal-dev acceptance/compare.jl
julia --project=~/Projects/Julia/universal-dev experiments/dynamic_router.jl
julia --project=~/Projects/Julia/universal-dev experiments/operation_axis.jl
```
