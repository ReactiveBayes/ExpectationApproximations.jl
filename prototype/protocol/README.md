# Prototype of the expectation protocol (U0.2)

The protocol was prototyped here (PLAN.md, milestone U0.2). It now lives in BayesBase (U1,
`src/expectations/`), and its backends in this package (U2). The prototype's packages,
`ProtoBase` and `ProtoBackends`, and the experiments on the router's cost and the operation axis
are in the history of this branch; `RESULTS.md` records what they measured.

What remains is the acceptance test, which runs two notebooks of the Universal Inference section
with their hand-written fallback rewritten on the protocol and this package's backends, and
compares the posteriors with the originals:

```sh
julia --project=~/Projects/Julia/universal-dev acceptance/compare.jl
```

`acceptance/free_energy.jl` checks U3 end to end: the mixture notebook's model gets a free
energy through the rule fallback's average energy and an expectation backend, `EADefault()`:

```sh
julia --project=~/Projects/Julia/universal-dev acceptance/free_energy.jl
```

They move into the universal rules package with U4.
