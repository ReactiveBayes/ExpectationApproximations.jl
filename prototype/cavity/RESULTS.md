# What the prototype of design B found

Design B works on all three models. On each it is more accurate than today's fallback, and on
the pendulum it smooths, which today's fallback can't. It also found four things the
ReactiveMP stage must do differently from what the design said, listed below. Numbers are
from `checks.jl` and `models/*.jl`, run in `universal-dev` on Julia 1.13.

## The engine against exact answers (`checks.jl`)

- **A linear-Gaussian chain** (T = 30) whose observations are raw log-densities:
  - Laplace propagation (LP), EP and CVI each give the Kalman smoother to 1e-15;
  - the free energy is −log p(y) to 1e-10;
  - every site equals its message (property 1 of the design);
  - each takes 2 sweeps.
- **The same chain with closed observations** performs no projection and is exact.
- **One variable with two raw edges** (two counts under a Gaussian prior):
  - LP reaches the Laplace approximation of the posterior to 5e-11, in 5 iterations;
  - EP reaches its fixed point to 1e-12 in 9 iterations, and with damping β = 0.5 in 44.
  - The loop among raw sites converges without damping.

## The three models

### Stochastic volatility (T = 200)

LP's fixed point is the global Laplace approximation, to 7.5e-10 in the means and sds (design
section 3).

Against the notebook's elliptical slice sampler:

| | RMSE of the means | RMSE of the sds | sweeps | parallel iterations |
|---|---|---|---|---|
| today, `fallback = LaplaceApproximation()` | 0.177 | 0.019 | | |
| global Laplace | 0.100 | 0.011 | | |
| LP | 0.100 | 0.011 | 4 | 264 |
| EP | **0.016** | 0.010 | 5 | 265 |
| CVI | **0.016** | 0.011 | 5 | 265 |

### The count series (T = 150)

Against the grid forward-backward:

| | RMS Δmean | max Δsd | free energy |
|---|---|---|---|
| today, Laplace | 0.112 | | 207.35 |
| today, CVI | 0.028 | | 207.74 |
| LP | 0.073 | 0.004 | 206.684 |
| EP | **< 1e-4** | 0.003 | 206.686 |
| CVI | 0.0002 | 0.006 | 206.737 |
| exact, −log p(y) | | | 206.635 |

- **The beliefs agree.**
  - Each raw edge's projection against its final cavity is the marginal.
  - Each transition's joint belief has the variables' marginals as its own, to 2e-9 at a 1e-6
    stopping rule.
  - These are the inconsistencies today's per-product projections have.
- **The free energy** of every scheme is within 0.1 of the exact one; today's is 0.7–1.1 above.

### The pendulum, smoothed (T = 100)

The reference is a particle smoother: forward filtering with 5000 particles, then 500
trajectories by backward simulation. Its own RMSE against the truth is 0.145, and its ±2 sd
coverage is 0.82.

| | RMSE | RMS Δ from the particle smoother | coverage | sweeps | time |
|---|---|---|---|---|---|
| EP | 0.143 | **0.016** | 0.84 | 12 | 3 s |
| EP, forward sites (O1) | 0.143 | **0.014** | 0.83 | 13 | 2 s |
| CVI | 0.143 | 0.016 | 0.84 | does not settle below 2e-3 | 42 s |
| LP | 4.39 | 4.27 | 0.20 | does not settle | 7 s |
| EP, points on the cavity | diverges | | | | |
| RxInfer, unscented by hand, 10 iterations | 0.247 | 0.159 | 0.62 | | |
| RxInfer, `fallback = CVI()` | fails: "`LaplaceForm` cannot tell the dimension" | | | | |

- **EP's first forward pass is the EP filter.** It matches a direct implementation with the same
  projection to 4 digits,
  0.073 RMS from the particle filter.
- **Its first backward pass already smooths,** 0.038 from the particle smoother; further sweeps
  bring it to 0.016.
- **LP is lost at the turn back across −π/2,** as the Laplace filter of the notebook was.
  Laplace's method picks one mode of a belief with two.

## What it found for the design

1. **EP's quadrature must be centred on the tilted density, not on the cavity.**
   - The message pulled back through the pendulum's transition is far sharper than the cavity:
     Q's angle variance is 2e-5.
   - Gauss–Hermite points on the cavity can't resolve it. The moments collapse, a site's
     precision reaches 1e36, and the run diverges ("EP, points on the cavity").
   - Points on the tilted density's Laplace approximation, weighted by the tilted density
     against it, fix it everywhere.
   - EA's `tilted_mean_cov` centres on `q`. The site policy needs the tilted-centred variant,
     which is an EA addition.
2. **The stopping rule must sit above the projection's own precision.** The marginals stop
   changing only to the precision of their projection:
   - LP: about 5e-9, set by Newton's tolerance;
   - CVI: about 1e-7 on stochastic volatility, and 2e-3 on the pendulum, where its sites keep
     moving although its means are as good as EP's.

   A stopping rule tighter than the projection's precision never fires. The engine's
   convergence check, and RxInfer's, must take a tolerance relative to the policy's.
3. **The schedule matters for the number of iterations, not for the answer.**
   - A parallel schedule, where every site is computed from the previous iteration, needs O(T)
     iterations on a chain: 264 on SV and 524 on the pendulum. Sweeps need 4–12.
   - In ReactiveMP a message is sent as soon as its inputs arrive, so one iteration propagates
     along the chain like a sweep.
   - A raw message must be computed from up-to-date cavities. The prototype's first pendulum
     run visited `x_t` before `s_t`, so it used a stale observation message and filtered the
     wrong way.
   - The reactive streams order this by dependency. This supports the design's stream
     `site_k = combineLatestUpdates(input_k, output_k)` over a fixed visiting order.
4. **Singular messages are ordinary, and indefinite ones are harmless where they go.**
   - The observation of `sin θ` says nothing about the velocity, so its site and many outbound
     messages are singular. The test for a proper cavity must tell singular from indefinite.
   - Under EP, outbound messages were indefinite 44 times. They went to Gaussian nodes, which
     handle them in natural parameters, and to Delta nodes, whose pull-back is a log-density.
     Nothing broke.
   - **Proper-ness is needed only where a projection needs a proper tilted density.**

## The open questions

- **O1, forward sites for Delta nodes:** they help a little, 0.014 against 0.016, at the same
  cost. They work only when the pushforward's site accepts an improper cavity, because the
  backward message is singular in the velocity.
  - **Recommendation:** keep today's moment-matched forward message as the default, and offer
    forward sites as an option later.
- **O2, an improper marginal:** with tilted-centred EP it never arose in the three models.
  Improper marginals came only with the failing cavity-centred projection, and clipping changed
  nothing.
  - **Decided (the user):** no proper-ness check at all. Improper messages are passed as
    computed, with nothing skipped or reported. The engine's skip, clip and counters were
    measurements for this question only.
- **O3, improper outbound messages:** they do occur, and must be allowed (finding 4).
  - ReactiveMP and ExponentialFamily must carry indefinite Gaussian messages in natural
    parameters to the nodes, as section 8 of the design says.
  - Rules that can't take them are a separate concern; the Gaussian rules can.
- **Damping:** none of the three models needed it. On the two-raw-edge toy it slowed
  convergence, 9 iterations against 44.

## The free energy in local terms

`local_free_energy` computes EP's free energy as ReactiveMP would, from terms each node and
variable computes alone. They are Bethe's own terms, except for nodes that send raw messages
(design section 9a).

- **The chain:** it equals the global form to 1e-10 under every policy.
- **The count series:** 1e-12 under EP, and 1e-9 under LP, its fixed-point tolerance.
- **The pendulum:** the global form needs every cavity's normaliser, and some cavities are
  improper, so it is undefined there. The local form needs none, and gives 37.27, against the
  particle filter's −log p(y) of 37.32 ± 0.02 (four runs, 50 000 particles).

A node's integral must be its scheme's own estimate. With an exact integral under LP the two
forms differ by 0.34 on the count series; with Laplace's they agree.

## A bug found on the way

EA's `gaussian_variational` failed for a one-element vector start: it took scalar quadrature
points. It was fixed in EA, with a test (`test/projections/variational_tests.jl`).
