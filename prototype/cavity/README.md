# Prototype of design B: sites at the variables (U7a)

A small factor-graph engine that computes a site per edge at each variable, `proj(m_k · c_k) /
c_k` against the cavity `c_k`, and every belief as a product of sites, as
`design/cavity_propagation.md` proposes for ReactiveMP. It changes no package. What it measured
is in `RESULTS.md`.

- `Cavity.jl`: the engine.
  - Gaussian sites in natural parameters, which may be indefinite.
  - Factors that send BP's messages: `Prior`, `LinearGaussian`, `Observation` (a log-density,
    always raw) and `Delta` (raw backwards, and forwards either moment-matched or, with
    `forward_site = true`, a raw pushforward).
  - `compute_site`, the one function a ReactiveMP stage would call, under the policies
    `LaplacePropagation`, `ExpectationPropagation` and `CVIPropagation`.
  - Two schedules, `:sweep` and `:parallel`.
  - The EP free energy in Bethe form.
- `checks.jl`: the engine against exact answers.
- `models/sv.jl`, `models/counts.jl`, `models/pendulum.jl`: the three models of the design's
  section 10.

Run each in the shared environment, from this folder:

```sh
julia --project=~/Projects/Julia/universal-dev checks.jl
julia --project=~/Projects/Julia/universal-dev models/sv.jl        # ~30 s, most of it the slice sampler
julia --project=~/Projects/Julia/universal-dev models/counts.jl    # ~10 s
julia --project=~/Projects/Julia/universal-dev models/pendulum.jl  # ~90 s
```

Each script prints `ok` or `FAILED` per criterion, and exits non-zero when one fails.
