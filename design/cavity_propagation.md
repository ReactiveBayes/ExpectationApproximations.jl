# Cavity propagation: sites at the variables (U7a, design B)

Status: a design, not implemented. The user chose design B, sites computed at the variables,
over design A, sites computed by the nodes. Its prototype on the hub is done, and it met every
criterion (section 10); the engine changes of section 9 come next. PLAN.md, U7a, links here.

## 1. The problem

Under the fallback today, a variable that receives a message with no closed form projects each
product of its messages separately. Take `x` with inbound messages `m_1, …, m_K`, where `m_3`
is a log-density from the fallback:

- the message out on edge 1 is `proj(m_2 · m_3 · … · m_K)`;
- the message out on edge 2 is `proj(m_1 · m_3 · … · m_K)`;
- the marginal is `proj(m_1 · m_2 · m_3 · … · m_K)`.

`m_3` is approximated K + 1 times, each time in a different context, and the fits disagree.
This has three consequences, all seen in U6:

- **The beliefs are inconsistent.** A marginal and the joint beliefs next to it are projected
  in different contexts, so they disagree, and the Bethe free energy is not a bound (the count
  series, PLAN.md "What the second round of U6 delivered and found").
- **A message may be projected without the context that makes it proper.** A message pulled back
  through a nonlinear function, say `sin(θ)`, says nothing about the velocity. Projected without
  the forward message, it has no Gaussian, so the pendulum cannot be smoothed.
- **Each projection handles every raw message at once,** which is the hardest version of the
  problem.

## 2. The idea: one site per edge

Each variable `x` has a family `Q`, the family of its beliefs (section 4). An inbound message is
**closed** when it is in `Q`, and **raw** otherwise: a log-density from the fallback, a
`GammaShapeLikelihood`, or a Gamma message to a variable whose `Q` is a Gaussian in `log x`.

The variable keeps one **site** `s_k ∈ Q` per edge:

- on a closed edge, the site is the message itself, `s_k = m_k`;
- on a raw edge, the site is
  ```
  c_k = ∏_{j ≠ k} s_j                      (the cavity: all the other sites)
  s_k = proj_Q(m_k · c_k) / c_k            (the site)
  ```

The variable then computes every belief from the sites, never from the raw messages:

- the message out on edge k is `∏_{j ≠ k} s_j`, which is the cavity `c_k`;
- the marginal is `∏_j s_j`, which equals `proj_Q(m_k · c_k)` for every raw edge k at a fixed
  point.

**The cavity of an edge is the message the variable sends out on it.** Today the variable
already computes it. What changes is that the variable uses it on that edge's inbound message
before multiplying.

`proj_Q` is any of the projections the fallback has:
- Laplace's method gives **Laplace propagation** (Smola, Vishwanathan and Eskin, 2004);
- moment matching gives **expectation propagation**;
- the variational Gaussian gives a CVI variant.

### Properties

1. **Exact where BP is exact.** If `m_k ∈ Q` then `m_k · c_k ∈ Q`, its projection is itself, and
   `s_k = m_k`. Exact rules and conjugate products don't change. A variable with only closed
   messages behaves as it does today.
2. **The beliefs agree.** Every message out and the marginal are products of sites in `Q`, so they
   are exact products of one family. They are one approximation seen from different edges.
3. **Each projection has one raw factor in it,** times a member of `Q`: `m_k · c_k`. For a
   Gaussian `Q`, that is one non-Gaussian factor times a proper Gaussian. The cavity gives the
   dimension and a proper base, which is what the pendulum's backward messages lack today.
4. **It is EP with a fully factorised site.** Take a factor `f(x, y)`. Under BP it sends the raw
   message `m_{f→x}(x) = ∫ f(x, y) c_y(y) dy`, where `c_y` is `y`'s cavity on that edge. The
   marginal of the tilted distribution `c_x(x) c_y(y) f(x, y)` in `x` is
   `c_x(x) · m_{f→x}(x)`. So `proj(m_{f→x} · c_x)` is EP's moment-matching step for
   `f ≈ s_x(x) s_y(y)` (Minka, 2001; Minka, 2005, "Divergence measures and message passing",
   section 4). For a factor of one variable, the observation's, it is EP exactly.

## 3. An example: a chain with three edges at `x_t`

```
   (transition t−1 → t) ──m_left──►  ( x_t )  ◄──m_right── (transition t → t+1)
                                         ▲
                                         │ m_obs
                                  (observation y_t)
```

Take Gaussian transitions with exact rules, and an observation node with no rules:
- `m_left` and `m_right` are closed: `s_left = m_left` and `s_right = m_right`;
- `m_obs` is raw: its cavity is `c_obs = m_left · m_right`, and its site is
  `s_obs = proj(m_obs · m_left · m_right) / (m_left · m_right)`;
- towards t+1 the variable sends `m_left · s_obs`, towards t−1 it sends `m_right · s_obs`, and
  towards the observation `m_left · m_right`.

On stochastic volatility, Laplace propagation's fixed point is the global Laplace
approximation, means and covariances. The sites are second-order Taylor expansions of each
`log p(y_t | x_t)` at the global mode, and only one term of the posterior's Hessian comes from
each observation. This makes it the first check of the prototype.

## 4. Families other than the Gaussian

Nothing above needs `Q` to be Gaussian. It needs `Q` to be an **exponential family**, where:

- a product is a sum of natural parameters, `η(a · b) = η(a) + η(b)`;
- a division is a difference, `η(s_k) = η(proj(m_k · c_k)) − η(c_k)`.

So a site is a member of `Q` in natural parameters, and products of sites and cavities stay in
`Q` exactly.

| Variable's domain | `Q` | What a site may be | Projections |
|---|---|---|---|
| real line, `ℝⁿ` | Gaussian | a Gaussian with any symmetric precision | Laplace, moment matching (EA's `tilted_mean_cov`), the variational Gaussian |
| positive | Gamma, or a Gaussian in `log x` | Gamma: `(α − 1, −β)` of either sign | moment matching (EFP's `ExpectationStrategy`); Laplace and the variational Gaussian in `log x` |
| `(0, 1)` | Beta, or a Gaussian in `logit x` | Beta: `(α − 1, β − 1)` of either sign | as above |
| simplex | Dirichlet | as Beta | moment matching |
| finite set | Categorical | — | products are already exact; nothing to do |

- **The site is only a factor and need not be a distribution.** Its natural parameters may lie
  outside the family's domain, for example as a negative precision or a negative `α − 1`. Only
  the cavity and the marginal must be proper (section 8).
- **`Q` comes from the variable,** from where the default projection already comes from (D33):
  - the variable's form constraint when it has one;
  - otherwise the domain its prior gives.

  Where exact messages are of one family, such as Gamma on a Poisson rate, that family is the
  right `Q`, because it keeps them closed: a Gaussian in `log λ` would make the Gamma prior a
  raw message.
- **Where B does not apply:** `Q` is not an exponential family, as with a `SampleList`, a
  mixture or a point mass. A site has no division then, and the variable projects each product
  as it does today. The two can coexist in one model, variable by variable.

## 5. Many edges, and many raw messages at one variable

With several raw edges, the sites depend on each other: `s_1` needs `c_1`, which contains
`s_2`, whose cavity contains `s_1`. That is EP's usual loop, and it is solved the usual way:

- **Initial sites.**
  - A raw edge starts from an initial site, by default the uniform one (`η = 0`).
  - Each cavity must be proper on its first use, so a raw edge's first site is computed only once
    its cavity is proper:
    - on a chain, the forward messages give that;
    - elsewhere, the user can supply an initial site through `@initialization`, as for an EP
      rule today.
- **The schedule.**
  - A site is recomputed when its raw message or its cavity changes, as EP revisits a factor
    when the rest of the approximation moves.
  - Inside an iteration, ReactiveMP's rule that a product waits for each of its inputs to update
    breaks the loop. It already iterates Probit's own-edge EP rule this way (PLAN.md, U7a).
- **Damping.**
  - `η(s_k) ← β η(s_k)_previous + (1 − β) η(s_k)_new`, in natural parameters.
  - This is ReactiveMP's `damp`, applied to sites rather than to outbound messages.
- **The cost.** It is one projection per raw edge per iteration. Today it is one per product, K
  + 1 per variable, each with every raw message in it. Each of the new projections is simpler:
  one raw factor times a proper member of `Q`.

## 6. Variables in the middle of the graph

The observation in section 3 is only an example of a raw message. A raw message can come from
any node, so variables with no observation get sites too:

- **A nonlinear transition.** The fallback's message backwards through `x_{t+1} ~ N(g(x_t), Q)`
  is `log ∫ N(x_{t+1} | g(x_t), Q) c(x_{t+1}) dx_{t+1}`, a log-density of `x_t`, so it is raw.
  At `x_t`, its cavity is everything else: the forward message and the observation's site. That
  is how the pendulum gets a smoother. Each backward step is a projection against a proper
  cavity, and the site that results is a Gaussian, possibly improper, which the exact rules of
  the Gaussian transition take as they are.
- **A Delta node `z = g(x)`.**
  - Its message to `x` is `c_z(g(x))`, the cavity of `z` pulled back. It is raw, and at `x` it gets
    a site against `x`'s cavity.
  - Its message to `z` is the pushforward of `x`'s cavity through `g`. Today the fallback
    moment-matches it without `z`'s cavity: this is the unscented filter's step.
  - **Open question (O1):** whether the forward direction should also get a site against `z`'s
    cavity. This would need the raw message as a lazy pushforward object `(c_x, g)`, whose
    moments under the cavity are weighted averages, `E_{c_x}[T(g(x)) c_z(g(x))] /
    E_{c_x}[c_z(g(x))]`. The prototype measures whether it matters.
- **A stochastic node with several random inputs.**
  - Each input `x` gets a site from the raw message `log ∫ f(x, y) c_y(y) dy` (section 2,
    property 4).
  - This must be BP's message, `log E_{c_y}[f]`. The fallback today sends `E_{c_y}[log f]`,
    which is the variational message. On a site, that would give mean field's answer under EP's
    name. Fixing it is a change to DMPR (section 9), useful under plain BP too.
- **A variable with only closed messages.** It does nothing new. On a linear-Gaussian stretch
  of a chain, the variables send exact messages as they do today, and the sites of the
  nonlinear stretches come in as Gaussians.

**What a node sees.** A node receives the variable's outbound message on its edge, which is that
edge's cavity, as before. Rules don't change, and neither do the fallback's arguments. Only the
variable's side changes.

## 7. B and A

Design A computes the same site in the node: a rule that reads the message on its own edge (the
cavity) and returns `proj(cavity · f) / cavity`, as Probit's EP rule does.

**What A can do that B cannot:**
- A node with several random inputs sees all their cavities, and can fit one joint `Q` over
  them, which is EP with structured sites.
- B fits one site per variable. For a factor of one variable, the two are identical.

**What A costs:**
- Rule matching must give a fallback its own edge without breaking exact rules (PLAN.md, U7a).
- Every kind of node the fallback serves would have to learn the scheme.

**B keeps A's door open.** A node with a hand-written own-edge rule sends a message in `Q`, which
B passes as closed. A node that needs joint sites can get an A-style rule later, alone.

## 8. Improper sites, cavities and marginals

- **Sites may be improper.**
  - A Gaussian site may have a singular or indefinite precision. The pendulum's `sin θ`
    observation gives precision near zero along the velocity, and a negative one where `sin` is
    concave against the cavity.
  - They are stored in natural parameters, for example as `NormalWeightedMeanPrecision` and
    `MvNormalWeightedMeanPrecision` with an unconstrained precision.
  - ExponentialFamily's `ExponentialFamilyDistribution(T, η)` checks `isproper` and errors,
    unless it is built with `attributes = nothing`.
  - BayesBase and ExponentialFamily need the operations on improper members:
    - the product, with `GenericProd` taking improper factors;
    - the division;
    - damping (`damp` errors on improper natural parameters today, PLAN.md U7a).
- **Nothing checks proper-ness (the user's decision, O2).**
  - Improper sites, cavities, outbound messages and marginals are allowed, and each is passed as
    computed. No update is skipped, clipped or reported for it.
  - There is no reliable automatic test for proper-ness, and the prototype found improper
    messages harmless where they went (O3).
  - A computation that can't use an improper input, such as the moments of an improper
    marginal, gives whatever it gives.
- **Code on the site path works in natural parameters,** through `weightedmean_precision`. It
  never calls `mean_cov`, `cov` or `logpdf` on a message that may be improper.
  - For an indefinite precision, FastCholesky's GMW81 fallback silently inverts a different,
    positive definite matrix.
  - The Gaussian rules that take such messages get natural-parameter methods
    (StandardMessagePassingRules).

## 9. What changes, package by package

### ReactiveMP: the site stage of a random variable

- **The place.**
  - `activate!(::RandomVariable, ::RandomVariableActivationOptions)` (`src/variables/random.jl`)
    feeds `input_messages` into the `EqualityChain` (the outbound messages) and into
    `collectLatest` (the marginal).
  - B puts a stage between them. Inbound message k, together with outbound message k, becomes
    site k, and the chain and the marginal take the sites.
- **The stream.**
  - `site_k = combineLatestUpdates(input_messages[k], output_messages[k])`, mapped through the
    site policy, with an initial site for raw edges.
  - `output_messages[k]` is already a stream of the chain, so no new kind of cycle is introduced:
    it is the loop of Probit's own-edge rule, kept inside the variable.
- **The interface.**
  - A new field of `RandomVariableActivationOptions`, `site_policy`. Its default, `nothing`,
    does today's thing: no stage, the raw messages go to the products.
  - The policy is called as `compute_site(policy, variable, message, cavity, previous) -> Message`.
    It returns the closed message unchanged (property 1), the new site, or `previous` when it
    skips (section 8).
  - ReactiveMP declares the function and knows no projection. DMPR implements it (below).
- **Damping:** the policy damps the site itself, so `DampingPostprocessor` stays for outbound
  messages.
- **Callbacks:** `BeforeSiteComputationEvent` and `AfterSiteComputationEvent`, with the
  variable, the edge, the message, the cavity and the site, as the product events do. The report
  (U8) takes its rows from them.
- **The free energy needs nothing from the site stage** (section 9a): the variables' Bethe term
  stays `(d_i − 1) H[q_i]`.
- **Tests:**
  - a variable with one raw edge reaches the fixed point in one step;
  - one with two raw edges reaches hand-computed EP;
  - closed edges pass unchanged;
  - `site_policy = nothing` changes nothing.

### BayesBase and ExponentialFamily: improper members

- Products and divisions of exponential-family members that may be improper, as natural
  parameters with no `isproper` check.
- Damping of them, with a test that a damped site is a natural-parameter mix.
- A cheap `isproper` for the cavity and marginal checks of section 8.

### DeltaMessagePassingRules: the site policy and BP's raw messages

- **`compute_site`** for the fallback's schemes. Each is `proj_Q(message · cavity)`, by the
  projection the scheme names, divided by the cavity:
  - `LaplaceForm` with the cavity as the base;
  - EA's `tilted_mean_cov` for a Gaussian `Q`;
  - EFP's `ExpectationStrategy` for the others;
  - the variational Gaussian for CVI.

  It reports an outcome (`converged`, `skipped`, `closed`) as the fallback does.
- **BP's raw message for stochastic nodes with several random inputs:** `log E_{c_y}[f(x, y)]`,
  by EA's expectations, instead of `E_{c_y}[log f]` (section 6). This is a fix to the BP
  fallback that stands on its own.
- **O1:** a lazy pushforward message for Delta nodes' forward direction, if the prototype shows
  it matters.

### RxInfer: the schemes

- **`LaplacePropagation()` and `ExpectationPropagation()`** (the names of D43) are fallback
  schemes, `infer(...; fallback = ExpectationPropagation())`.
  - Each sets the rule fallback, as today's schemes do.
  - Each also sets the site policy of every random variable whose `Q` is an exponential family.
- **Their options:**
  - `damping = β`;
  - the projection, for `ExpectationPropagation(projection = …)`: moment matching by default,
    with the variational Gaussian as the CVI variant;
  - `iterations` stays `infer`'s.
- **Initial sites** through `@initialization`, under a name to be decided, for example
  `site(x, :edge)`, or the initial messages it already has.
- **The free energy** under these schemes is EP's (Minka 2001, section 3.4), in the local terms
  of section 9a. Like Bethe's under projections, it isn't a bound. At a fixed point it
  approximates `−log p(y)`, and the notebooks compare it with the exact one.
- **The report** gets a column for sites: the edges given sites, the projection and the
  outcome.

### 9a. The free energy in local terms

EP's free energy, `−(Σ_a log Z_a − Σ_i (d_i − 1) log ∫ q_i)` with `Z_a = ∫ f_a ∏_i c_ia`, can
be split into terms that each node and each variable computes from what it sees. Bethe's terms
stay as they are except one kind:

- **A variable:** `(d_i − 1) H[q_i]`, as today.
- **A node with rules,** whose messages are closed: `E_b[−log f_a] − H[b_a]` with
  `b_a ∝ f_a ∏_i c_ia`, as today. At a fixed point the marginals of `b_a` are the `q_i`.
- **A node that sends a raw message** (the fallback's nodes):
  ```
  F_a = −log ∫ f_a ∏_i c_ia + Σ_i E_{q_i}[log c_ia]
  ```
  from its inbound messages `c_ia`, which are the cavities, and the marginals `q_i`.
  - Rescaling any `c_ia` doesn't change it, so the cavities enter as kernels,
    `log c(x) = ξᵀx − xᵀΛx/2`.
  - So no cavity has to be proper, only the tilted density `f_a ∏ c` must be integrable.
  - The integral is the scheme's own estimate: Laplace's for `LaplacePropagation()`, the
    moments' quadrature for `ExpectationPropagation()`.

Why it holds:
- Sites multiply to the marginal, so `Σ_a η(c_ia) = (d_i − 1) η(q_i)`. The variable term is then
  `(d_i − 1) A(q_i) − Σ_a E_{q_i}[log c_ia]`.
- A closed node's `−log Z_a` equals its Bethe term minus `Σ_i E_{q_i}[log c_ia]`.
- Moving the `E_{q_i}[log c_ia]` of every node to its own term leaves the variables' and closed
  nodes' terms unchanged.

The prototype checks it against the global form:
- 1e-12 under EP and 1e-9 under LP (its fixed-point tolerance), on the chain and the count series;
- on the pendulum, where improper cavities make the global form undefined, the local form gives
  37.27 against a particle filter's −log p(y) of 37.32 ± 0.02.

In ReactiveMP this is a fallback energy that sees the node's inbound messages as well as the
marginals, under a propagation scheme. Nothing else in the scores changes.

### What does not change

- Rules, rule matching and the fallback's arguments.
- MessagePassingRulesBase.
- Hand-written EP rules such as Probit's, whose messages are closed.

## 10. The prototype (done)

Before any engine change, the sites were computed on the hub, in `prototype/cavity/`, by a
small factor-graph engine:
- the raw messages are written as BP defines them, the log-densities the fallback sends;
- the sites come from EA's projections;
- the products and divisions are done in natural parameters.

The prototype runs three models, each with a criterion it must meet:

| Model | Reference | Criterion |
|---|---|---|
| Stochastic volatility, Laplace propagation | the global Laplace approximation (EA's `laplace` on the joint) | means and covariances to 1e-6 (section 3) |
| The count series, EP and Laplace propagation | the grid forward-backward of the U6 notebook | the beliefs agree on every edge; the means closer to the grid than today's (RMS 0.112 Laplace, 0.028 CVI); the EP free energy against −log p(y) |
| The pendulum, smoothed | a particle smoother (forward-filtering backward-simulation) | it runs, against today's filter that can't smooth; the RMS difference from the particle smoother |

**What it found** (`prototype/cavity/RESULTS.md` has the numbers): every criterion is met.

**The models.**
- **Stochastic volatility:** LP equals the global Laplace approximation to 7.5e-10. EP's RMSE
  of the means against the sampler is 0.016, against today's 0.177.
- **The count series:**
  - EP's means are within 1e-4 of the grid, against today's 0.028;
  - its free energy is 206.686, against the exact 206.635 and today's 207.35;
  - the beliefs agree.
- **The pendulum smooths:**
  - EP is 0.016 RMS from the particle smoother, with coverage 0.84 against the smoother's 0.82,
    in 12 sweeps;
  - RxInfer's hand-chosen unscented smoother is 0.159 from it;
  - LP is lost at the turn, as the Laplace filter is.

**Changes to the design.**
- **EP's quadrature is centred on the tilted density** (its Laplace approximation, weighted
  against it), not on the cavity. On the cavity it diverges where a factor is much sharper than
  its cavity. Section 9's DMPR policy needs this, through a tilted-centred variant of EA's
  `tilted_mean_cov`.
- **The stopping rule sits above the projection's precision:** about 5e-9 for Laplace, and up
  to 2e-3 for CVI on the pendulum.
- **The test for a proper cavity tells singular from indefinite.** A singular message is
  ordinary: an observation of `sin θ` says nothing about the velocity.
- **The schedule.** A raw message must come from up-to-date cavities, which the stream
  `combineLatestUpdates` of section 9 gives. A parallel schedule needs O(T) iterations on a
  chain, and sweeps need 4–12.

## 11. Open questions

- **O1.** Should a Delta node's forward message get a site against the output's cavity, through a
  lazy pushforward? (section 6)
  - *Prototype:* forward sites help a little (0.014 against 0.016 on the pendulum), and only
    once a pushforward's site may use an improper cavity.
  - Keep today's moment matching as the default, and add forward sites as an option.
- ~~**O2.**~~ When a site would make a marginal improper, should the default skip it or clip it?
  (section 8)
  - *Decided:* neither. Nothing checks proper-ness, and improper messages are passed as
    computed. In the prototype, improper marginals never arose with tilted-centred EP, and
    clipping changed nothing.
- **O3.** May an outbound message be improper, for nodes whose rules accept it? (section 8)
  - *Prototype:* yes, it must be allowed. Under EP, outbound messages were indefinite 44 times
    on the pendulum, sent to Gaussian and Delta nodes, and nothing broke.
  - Proper-ness is needed only where a projection needs a proper tilted density.
- **O4.** What does `@initialization` call an initial site?
- **O5.** Which projections does `ExpectationPropagation()` offer, besides moment matching (the
  variational Gaussian for the CVI variant; Laplace being `LaplacePropagation()`)?

## References

- T. P. Minka (2001). Expectation propagation for approximate Bayesian inference. *UAI*.
- T. P. Minka (2005). Divergence measures and message passing. Microsoft Research
  MSR-TR-2005-173.
- A. J. Smola, S. V. N. Vishwanathan and E. Eskin (2004). Laplace propagation. *NeurIPS*.
- A. Vehtari et al. (2020). Expectation propagation as a way of life. *JMLR* 21(17), on site
  initialisation, damping and improper sites.
