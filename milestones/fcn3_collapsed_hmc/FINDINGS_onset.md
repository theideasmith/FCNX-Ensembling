# FCN3 onset of learnability at fixed κ, and where theory and sampler part ways (2026-11-10)

Setting: d = 150, N1 = N2 = 1500, χ = 1, κ = 0.015, σ_A² = 8/1500, σ_B = 1.2554, σ_w² = 0.5, ε = −0.3√6,
P = 750 … 48000, prior start, three datasets per P (`queue_onset.sh`, group `onset_d150`, full networks in `nets/`).
Plots: `plot_onset.py` → `plots/onset/onset_d150_learnability.png`, `plots/onset/onset_d150_actions.png`.

## What agrees
- L1 at every P; the onset of L3 in shape (sampler 0.05–0.075 below the theory at every P).
- Read-in layer: inner well, its split into two wells from P ≈ 12000, the outer well moving in with P, and the number of
  outer read-in neurons up to P = 12000 (within 10–15 %).
- Hidden layer (h|He1, h|He3) from P = 750 to 12000, and h|He1 to 24000.

## Fixed along the way
- The theory's default start lands on a local maximum at P = 750 (far group of hidden neurons, no outer read-in well).
  Solving every P from several starts and keeping the largest W (`plot_onset.theory_all`, `theory_phase_check.py`)
  gives the outer-read-in state at P = 750 (W higher by 106), in agreement with the sampler. Elsewhere all starts agree.
- Theory hidden-layer marginals: the prior covariance G is nearly singular (cond ~ 1e11), which broke the first two
  methods; `plot_onset.hidden_marginals` moves solver particles along prior regression lines and matches the solver's
  own second moments to 1–2 %.
- σ_B² is handled identically in theory and sampler (J11, H11 from saved networks = sampler's record = theory's
  summary values).

## Ruled out
- The term the theory drops from the output-dual covariance, χ(σ_A²K + κ/P)⁻¹: per hidden neuron it costs 0.02–0.18
  nats and is flat in h (sd ≤ 0.012) — a constant shift (`cavity_decomposition.py`).
- The factor 1/(1 − ℓ) the theory drops from each hidden neuron's readout term, 1 − ℓ = posterior variance of that
  neuron's readout weight / its prior variance (ℓ = 0.04 → 0.31 from P = 750 to 48000): including
  it (`onsager_test.py`) raises L3 slightly (away from the sampler) and does not split h|He3.
- The theory's average over the perpendicular directions for a given hidden neuron, e_n = E_x0[phi_n erf(a(x0) /
  sqrt(1 + 2 q_h))] (one blur q_h at every x0), checked on every hidden neuron of saved networks against the exact
  population projections (fresh inputs; `perp_blur_test.py`): per-neuron error ~1e-3, mean output within 0.007 in
  every mode (P = 48000: L3 0.889 from the formula, 0.898 exact; P = 12000: 0.780 / 0.781), and the same in every
  region (large-h3 and far-He1 neurons included). The perpendicular variance does depend on x0 (ordinary neurons
  1.02 → 0.87 of its average from x0 = 0 to 3, large-h3 neurons 0.97 → 1.35), but using the true profile changes
  nothing visible. The map from (h, q_h) to the output is right; the differences are in the distribution of h.

## Main finding: the networks overfit their training set; the theory cannot
- Test MSE / training MSE = 2.5–3.5 on the onset line; the He3 residual on the training set is about half the
  population residual (`overfit_check.py`).
- Each hidden neuron is pulled by the TRAINING residual: its exact pull is 92–99 % correlated with the theory's form
  but only 0.44–0.93 of the size the population residual implies. The theory, which replaces the training loss by its
  population average, sets the two residuals equal.
- At fixed theory (κ ∝ P), the L3 gap shrinks with P: 0.087, 0.030, 0.017 at P = 3000, 6000, 12000 (dataset 2).
- Along the κ0 = 0.015 ray (`ray_overfit_check.py`) the train/test gap closes with d as the L3 gap does: test/train MSE 2.75, 1.67, 1.39 and
  He3 residual train/test 0.60, 0.76, 0.81 at d = 150, 300, 600.
- A fixed-feature readout cannot produce a 2× residual ratio (its degrees of freedom are ≤ N2 = 1500 ≪ P); the
  overfitting is in the learned features.

## Sampler-side caveats
- The number of outer read-in neurons is not at equilibrium from P = 6000 on: at 6000 and 12000 it falls through the
  theory's value and is still falling at the last iteration; at 24000 it starts at half the theory's and creeps up;
  at 48000 it is fixed (54–63 vs 150). Almost none in the far read-in well (theory: 1.4 and 2.6 per network).
- Holding the theory at the runs' counts (`theory_fixed_count.py`, `theory_fixed_count_hidden.py`) lowers its L3 by
  at most 0.009 and moves its hidden layer away from the sampler's: the frozen count is not the cause of either.
- In the hidden layer the far He1 group (1–3 neurons) and the outer He3 group (17 neurons at P = 48000) are also
  frozen (same neurons for the whole run). The inner h|He3 double well is equilibrium (neurons switch sign freely).
- Read-in kernels from the saved networks are 10–12 % weaker than the theory's at P = 3000–12000 and 26–38 % weaker at
  24000–48000 (`empirical_kernel_test.py`; its hidden-layer solves at P = 12000 and 24000 are not self-consistent and
  should not be used).

## Wide range in the hidden layer, and the large-h|He3 group at P = 48000
- The theory has no further hidden-layer wells: computed out to |h1| = 8 and |h3| = 3 (`plot_hidden_wide.py` →
  `plots/onset/onset_d150_hidden_wide.png`), the action only rises beyond the wells already shown.
- At P = 24000 the sampler's points at |h3| ≈ 0.2–0.28 are the far He1 group (|h1| 1.85–2.55) seen in He3; the
  theory's kinks at |h3| ≈ 0.2 are its far-He1 shoulder seen the same way.
- At P = 48000 there is a separate group (`outer_he3_group.py`): in every dataset 17 hidden neurons with |h3| ≈ 0.43,
  |h1| ≈ 0.37, the same neurons for the whole run. All of their h3 comes from the ~55 outer read-in neurons, onto which
  they put weights 4–5 times the prior size. Their exact fit gain is 470–535 nats (ordinary neurons 12–15). About half
  of their pull comes through their dependence on input directions other than x0 (ordinary neurons: none). The
  theory's gain at the same neurons is 225–250 against its prior cost of 420–460, so the theory has no well there; it
  has no channel for a hidden neuron's dependence on the other input directions to correlate with the residual.
- A higher x0 tower does not help: with the tower at He11, 21, 31 or 41 the other-direction share of these neurons'
  pull stays 0.48–0.57 (`outer_he3_tower.py`).
- That part is not overfitting (`outer_he3_fresh.py`): it comes from the other-direction part of the network's own
  mean output (it pulls with the same sign as the x0 part) and is as large or larger on a fresh dataset; the
  training-only correlation with the x0 residual is 0–2 nats. These neurons cancel the dependence on the other input
  directions that the few outer read-in neurons, shared by all hidden neurons, put into the output. The theory treats
  each hidden neuron's dependence on those directions as independent noise, so it cannot produce them.
- For ordinary neurons the same split shows the overfitting: on the training set their other-direction part
  correlates with the x0 residual (22–29 nats, 1–2 on fresh data), cancelled by the output's other-direction part.

## Perpendicular directions in the hidden layer (`plot_perp_projections.py`)
- Projections of the hidden pre-activations on He1, He3, He5 of each perpendicular coordinate shrink uniformly with P:
  mean square / untrained 0.98 at P = 750 down to 0.90, 0.85, 0.81 (He1, He3, He5) at P = 48000. No single
  perpendicular direction stands out from P = 3000 on (strongest direction as in untrained networks).
- At P = 750 and 1500 (training set at or below the width) the hidden layer does pick out one perpendicular direction:
  it overlaps 0.4–0.66 (P = 750) and 0.13–0.45 (P = 1500) with the training set's sum_mu y_mu x_perp,mu and
  sum_mu y_mu He2(x0_mu) x_perp,mu, against ~0.07 for a random direction, falling to that level by P = 6000. In one
  P = 750 dataset several read-in neurons sit at |w . u| ≈ 0.35–0.42 along it (untrained maximum ≈ 0.21–0.26).
  These directions are zero in the population, so this is learning of the training set, not of the target.

## The hidden-layer equation (E2) on its own
Fed measured inputs open loop (`hidden_equation_test.py`, `hidden_equation_test2.py`), (E2) is extremely sensitive to
the force: 10–20 % changes in t switch it between one well, several wells and a blow-up. So the large-P differences
in the hidden-layer shape are amplified consequences of the residual and read-in differences and cannot by themselves
show that (E2) is wrong.

## Fixed theory, more data (`queue_fixedtheory.sh`, dataset 0)
Theory held at an onset point while P grows with κ ∝ P:

| theory point (L3 theory) | P = 1× (κ 0.015) | 2× (κ 0.03) | 4× (κ 0.06) | outer read-in neurons (theory) |
|---|---|---|---|---|
| B, P = 24000 (0.913) | 0.839 | 0.874 | 0.876 | 47, 55, 60 (108) |
| A, P = 12000 (0.855) | 0.788 | 0.817 | 0.834 | 72, 96, 114 (78) |

At A the gap keeps closing (0.068, 0.038, 0.021); at B it halves and then stops (0.074, 0.039, 0.037). The outer
read-in count is again fixed from burn-in on in every run.

Theory held at each run's outer read-in count (`theory_fixed_count.py` → `plots/onset_checks/theory_fixed_count_d150.png`;
read-in prior tilted by exp(mu 1[|w0| > barrier]), mu scanned): the count is stiff in the theory (20 nats per neuron
moves it 15–25 %, the outer well moving out to compensate) and at the runs' counts its L3 is at most 0.009 lower. The
gaps stay (0.035–0.08 on the P scan, 0.027–0.042 for κ ∝ P), so the frozen count is not the cause.

## Weights between the layers: nothing beyond each hidden neuron's own h (2026-11-11, `coupling_width.py`)
- Hidden neurons with large |h|He3| put large weights onto the outer read-in neurons (correlation 0.63–0.68 at
  P = 750, 0.84–0.85 at P = 48000, where the 17 large-h3 neurons have rms weight 2.5–3 prior units), and the outer
  read-in neurons' outgoing columns line up (cosine with the other outer ones' sum 0.34–0.38 at P = 750, 0.41–0.44 at
  P = 48000; inner neurons ~0 or −0.08).
- All of it is what independent hidden neurons give when each one's weights are drawn from the prior given its own
  projections on He1..He11 of x0 (what the theory assumes): the same statistics to two or three decimals at
  P = 750, 3000 and 48000, every dataset. The part of the weights those projections do not fix has the singular
  values of prior noise, and for the 17 neurons rms 0.98–1.01 onto the outer read-in neurons. So this correlation is
  not a coupling beyond the theory and not by itself evidence of glassiness; it is why one read-in neuron cannot move
  without shifting the h of every hidden neuron that weights it.
- Width test (`queue_wide.sh`, `plots/width/coupling_width_P750.png`): N1 = N2 = 1500, 3000, 6000 (χ = N/1500, the
  same theory) at P = 750, three datasets each. L3 per dataset is the same at every width (0.198/0.199/0.200,
  0.218/0.220/0.220, 0.211/0.210/0.210; theory 0.294), outer read-in share 0.8–1.1 % at every width (theory 1.12 %),
  and the read-in action (with its shelf between the wells), h|He1 and h|He3 (narrower than the theory's) lie on top of
  each other. The coupling statistics fall with width (correlation 0.66 → 0.52 → 0.41, column cosine 0.36 → 0.26 →
  0.19), and the independent-neuron comparison falls with them to the third decimal: the same h is spread over more
  outer read-in neurons. So at this P the theory–sampler differences do not come from finite width.

## Dependence on the other input directions at P = 750 does not shrink with width (`width_perp.py`)
- The other-direction part of the network's mean output (f_bar minus its x0 tower to He21) carries 4.5 %, 9.4 %,
  4.6 % of f_bar's mean square (datasets 0, 1, 2) at N = 6000, against 4.9 %, 9.8 %, 4.8 % at N = 1500; about the same on a fresh
  dataset. At P = 48000 it was 0.6 %. The theory sets it to zero.
- The hidden layer projects onto one perpendicular direction more than an untrained network (top singular value
  1.05–1.15 × untrained at N = 1500, 1.16–1.27 × at N = 6000; the second stays at 1.00), and that direction overlaps
  0.56–0.66 (N = 1500) and 0.84–0.88 (N = 6000) with the training set's sum_mu (y − f_bar)_mu x_perp,mu (a random
  direction: ~0.08). Each hidden neuron's tilt toward it is small, but all hidden neurons share it, so the readout adds
  them up and the output's other-direction part does not average away as N grows.
- Readout term of each hidden neuron on the training set, through x0 / through its other-direction part: median
  0.33–0.57 / 0.12–0.40 nats, and for the 2 % with the largest |h|He3| 4.7–7.6 / 0.5–3.3 nats; the same at every width.
- So the dependence on the other input directions here is fitting chance correlations between the targets and those
  directions in a training set of P = 750 = 5 d: a finite-P effect, which a wider network does not remove.
- The outer read-in neurons are a group pointing the same way (`readin_align_width.py`): 12–16, 27–33, 56–64 of them
  at N = 1500, 3000, 6000 (about 1 % at every width), with mean pairwise cosine between their sign-aligned
  perpendicular weight directions 0.18–0.32 at every width (inner neurons 0.0006). Each overlaps 0.35–0.52 with the
  training set's sum_mu (y − f_bar)_mu x_perp,mu (unrelated directions 0.065, inner neurons 0.08), the same at every
  width, and their common direction overlaps it 0.69–0.79 at N = 1500 and 0.82–0.91 at N = 6000. This is the case of
  `notes/perpendicular_term_fcn2_fcn3.tex` where the overlap between hidden units through the shared read-in
  functions is of order one rather than N1^(-1/2): the group grows with N and each member's tilt does not shrink.
  Dataset 1 has the largest tilt at every width (pairwise cosine 0.31–0.32) and the largest other-direction output
  share (9–10 %): the tilt follows the training set.
- The pairwise cosine equals the square of each neuron's cosine with the group's common direction (N = 6000: 0.188 vs
  0.418² = 0.175, 0.321 vs 0.557² = 0.310, 0.197 vs 0.431² = 0.186; the same at N = 1500 and 3000), which is what
  neurons that each tilt independently toward one shared direction give. So this is not neurons coupling to each other: it is
  a field common to all of them (the training set's residual in the other directions), which breaks the isotropy of
  the perpendicular weights the theory assumes. Population averaging removes that field; large N does not.

## Overlap between hidden units through the shared read-in, by degree in x_perp (`hidden_overlap_width.py`)
- rms cosine between two hidden units' perpendicular fields, P = 750, mean of three datasets, trained / untrained:
  degree ≥ 2: 0.0268 / 0.0264 (N = 1500), 0.0190 / 0.0187 (3000), 0.0137 / 0.0133 (6000), ratio 1500 → 6000 0.51
  (N^-1/2: 0.50), as `notes/perpendicular_term_fcn2_fcn3.tex` states for independent rows. Degree 1: 0.086, 0.084,
  0.083, the same as untrained, ≈ 1/sqrt(149): set by d, not N.
- Among the 2 % of hidden units with the largest |h|He3|: degree ≥ 2 overlap 0.042, 0.030, 0.024 (ratio 0.58),
  above the rest at every width. In dataset 1 (the most tilted outer read-in group) it is 0.051, 0.038, 0.036 and
  nearly stops falling between N = 3000 and 6000.

## Outer read-in neurons at large P: their perpendicular directions cancel (2026-11-12, `readin_align_P.py`)
Size of the outer neurons' summed perpendicular direction, |sum_j s_j u_j| / sqrt(n_out) (s_j = sign w0_j; random
directions: 1), every saved network of the onset runs:

| P | 750 | 3000 | 12000 | 24000 | 48000 |
|---|---|---|---|---|---|
| ratio | 2.00 | 1.63 | 0.84 | 0.51 | 0.41 |

- At P = 750 and 3000 they share a direction (the training set's residual direction, as above); by 12000 that is gone.
- At P = 24000 and 48000 they spread out more evenly than random directions: mean pairwise cosine −0.013 to −0.017
  (−0.019 is the most negative possible for ~54 vectors), rms pairwise cosine 0.070 against 0.082 for random
  directions (this one needs no sign convention), no overlap with the training set's directions. Inner neurons look random.
- Each outer neuron's perpendicular direction and outgoing weight column decorrelate within 50 iterations, while the
  set of outer neurons does not change in 700. So the cancelling is maintained by the sampler, inside a frozen
  membership.
- Reading: with few outer neurons their summed perpendicular direction would put a perpendicular part of size
  ~1/sqrt(n_out) into the output; the network suppresses it by correlating the read-in neurons (and, from the section
  above, with the large-h3 hidden neurons). The theory's independent, isotropic read-in rows have neither the problem
  nor the fix.

## Which hidden neurons make which part of the output (2026-11-12, `he3_route.py`)
Each hidden region's contribution to the mean output's Hermite coefficients. Sampler: exact posterior-mean readout
times each neuron's population projections, every saved network. Theory: sigma_A^2 E[e (t.e) 1_region] over the
solver's own hidden-field particles (they reproduce its L3 to 0.007). Regions: far He1 |h1| > 0.8; large h3
|h3| > 0.25 and |h1| <= 0.8; ordinary = the rest. P = 48000 (target He1 = 1, He3 = −0.735, rest 0):

| | neurons/network | He1 | He3 | He5 | He7 |
|---|---|---|---|---|---|
| theory, ordinary | 1488 | 1.088 | −0.723 | +0.112 | +0.012 |
| theory, far He1 | 12.5 | −0.094 | +0.029 | −0.015 | +0.009 |
| sampler, ordinary | 1481 | 0.863 | −0.813 | +0.273 | −0.098 |
| sampler, large h3 | 17 | +0.172 | +0.150 | −0.138 | +0.094 |
| sampler, far He1 | 2 | −0.027 | +0.010 | −0.006 | +0.004 |

- Neither outer group produces the output's He3; both correct what the ordinary neurons produce.
- Sampler: ordinary neurons overshoot He3 (alone they would give L3 = 1.11) and produce a large unwanted He5 and He7.
  The 17 large-h3 neurons remove half the He5 and all the He7 and in doing so take back 0.150 of He3 (L3 −0.20, to
  0.889); they also supply the He1 the ordinary neurons fall short of.
- Theory: ordinary neurons overshoot He1 instead (1.088) and make less He5. Its force t is dominated by the He5
  residual (t1, t3, t5 = −8e3, −1.2e5, −3.1e5). Its 12 far-He1 neurons trim the He1 overshoot and a little He5.
- So the theory's He1 outer well is too deep and its large-h3 well missing because its ordinary neurons produce a
  different output to correct (too much He1, too little He3, He5, He7). The sampler's ordinary neurons carry more h3
  and less h1 (central sd h3 0.079 against 0.062, h1 0.144 against 0.172).
- At P = 24000 and 12000 the ordinary neurons produce everything in both; the far-He1 group (sampler 1.7 per network)
  subtracts 2-3 % of He1 and He3.

## The perpendicular part of the output, measured, and a surrogate theory that includes it (2026-11-12)
Measured on 9 saved networks per P (`measure_perp_pull.py`, population averages over 2^19 fresh inputs):
- f_bar_perp (mean output minus its x0 tower He1..He11) has rms 0.155 at P = 24000 and 0.098 at 48000, as large as
  the He3 and He5 residuals. Only 2-30 % of it is a radial function (He_n(x0) times |x_perp|^2), so it is not the
  theory's use of the mean |x_perp|^2.
- Each neuron's own perpendicular output summed incoherently would give 1.4x (24000) and 4x (48000) the measured
  |f_bar_perp|^2: the network already cancels most of it. 87-99 % of it comes from the ordinary neurons
  (|e_perp|^2 ~ 0.32 each).
- Each hidden neuron's overlap with it, c_k = <t_perp | erf o h_k> (t_perp = -f_bar_perp / lambda), is mostly a
  linear function of its x0 projections: c ~ gamma . e, R^2 0.80-0.87 overall, 0.98 for the large-h3 group; only the
  He1 and He3 components of gamma are stable across networks. For ordinary neurons c is anticorrelated with the x0
  pull s = t . e (−0.93), for the large-h3 group correlated (+0.94).
- The exact readout means follow s + c better than s at P = 48000 (R^2 0.88 against 0.82), not at 24000 (0.51 against
  0.97); their size is 0.4-0.74 of the theory's sigma_A^2/N2 (t . e).

Surrogate (`perp_surrogate.py`): with c = gamma . e the pull becomes 1/2 g~ ((t + gamma) . e)^2 and the readout
condition (sigma_A^2 K + lambda)(t + gamma) = y + lambda gamma, i.e. the theory with target y + lambda gamma. Measured
shift on (He1, He3): (0.0064, 0.047) at P = 24000, (0.0032, 0.018) at 48000 (the He3 target gets smaller).

| P = 48000 | L3 | f_bar He5 | far He1 /net | large h3 /net | sd h1 | sd h3 | outer read-in |
|---|---|---|---|---|---|---|---|
| theory | 0.947 | 0.097 | 11.6 | 0 | 0.218 | 0.063 | 150 |
| surrogate (shift x1) | 0.923 | 0.096 | 0.66 | 0 | 0.170 | 0.060 | 145 |
| surrogate (shift x2) | 0.899 | 0.095 | 0.44 | 0 | 0.168 | 0.059 | 141 |
| sampler | 0.899 | 0.123 | 2.0 | 17 | 0.164 | 0.093 | 57 |

P = 24000: L3 0.913 (theory), 0.851 (shift x1), 0.788 (x2), sampler 0.839; no far-He1 or large-h3 group in any
theory version (sampler 1.7 far He1); outer read-in 108, 100, 93 against 50.
- The L3 change follows the He3 shift almost one for one (shift / |y3| = 0.064 and 0.024), so the L3 agreement is
  largely put in by hand.
- Not put in by hand: at P = 48000 a 2.4 % smaller He3 target removes the theory's far-He1 group (11.6 → 0.66 per
  network, sampler 2.0) and narrows h1 to the sampler's width (0.218 → 0.170, sampler 0.164).
- Not reproduced: the large-h3 group, the h3 width, the outer read-in count, the He5 output. The full six-mode shift
  (noisy) gives the same (L3 0.925, far He1 0.61, no large-h3 group). So the part of the perpendicular pull that is a
  linear function of a neuron's x0 projections does not create the large-h3 group.

## Term by term: which part of the hidden-neuron action keeps the theory from the large-h3 group (`action_terms.py`)
(E2) split into prior on h (He1-He3 block; He5..He11 given them), prior on q_h, x0 gain 1/2 g~ (t.e)^2, and the
networks' perpendicular gain 1/2 g~ [(t.e + c)^2 - (t.e)^2]; each evaluated with the theory's ingredients (G, qbar, M,
t at its solution) and the sampler's (each network's own J_hat, qbar, M, population residual t, measured c), at the
same states. Large-h3 group minus ordinary neurons, P = 48000, nats (negative favours the group):

| term | theory's ingredients | sampler's ingredients |
|---|---|---|
| prior on h, He1-He3 block | +431 | +336 |
| prior on q_h | +46 | +50 |
| x0 gain | −225 | −89 |
| perpendicular gain | (absent) | −640 |
| sum | +252 | −343 |

- He5..He11 given He1, He3: each G pins them to its own line (conditional variance ~1e-10), and the two lines differ
  (theory's h7..h11 1.4-3x larger, from its 1.4-2.6 read-in neurons per network beyond |w0| = 1.2; sampler 0-0.33).
  Evaluated on the other's line this costs 7e3-2.5e4 nats, but it does not decide anything: a large-h3 neuron put on
  the theory's line has the same gains (x0 −228 vs −225, perpendicular −632 vs −640).
- He1-He3 prior: the theory charges ~95 nats more, from its stronger He1-He3 correlation in G (−0.79 against −0.73;
  large-h3 neurons have h1 and h3 of the same sign, which that correlation penalises).
- Without the perpendicular gain the two sums agree (+252 / +297). The perpendicular gain is the term that flips the
  sign; ~2/3 of it is the cross term 2 (t.e)(gamma.e), present with the theory's t as well (−666 with the theory's t).
- Why the surrogate (target shift) still had no large-h3 group: with c = gamma . e the perpendicular part is
  indistinguishable from a shift of t, and the readout condition (E4) re-solves the total pull vector u = t + gamma
  from the target, so it ends up close to the theory's t. In the networks the pull vector is much larger than the
  theory's for the same output (t + gamma: He3 −2.1e5, He5 −5.5e5 against the theory's −1.2e5, −3.1e5), because each
  neuron's readout is only 0.41-0.66 of sigma_A^2/N2 times its pull (overfitting: readouts follow the training
  residual, about half the population one). The theory ties pull and output together; the networks do not.
- Caveat: the sampler-ingredient sums use population residuals and are not calibrated occupancies (−343 would put far
  more than 17 neurons there); only the comparison between ingredient sets at the same states is meaningful.

## The theory with the networks' population residual held fixed (`fixed_field_continuation.py`)
Hidden-neuron action term -1/2 g~ (t . e + c)^2 with t . e + c replaced by u . e, u fixed: the theory is solved as
usual, then t is frozen in the same solver object and the read-in / hidden equations are solved for a stationary
point by plain Newton while u is moved from the theory's t (alpha 0) toward the networks' values (alpha 1).
P = 48000. Check: alpha 0 reproduces the theory (|g| 5e-4, far He1 11.55, outer 150, f_bar 1.003 / −0.696).
- u -> t_s + gamma (networks' population x0 residual plus c = gamma . e): no stationary point beyond alpha = 0.027.
- u -> t_s (x0 residual only): none beyond alpha = 0.0004.
- Solved from the prior instead, with u fixed (`fixed_field_surrogate.py`), every hidden neuron runs to large h;
  even u = the theory's own t does that, so with t frozen the theory's equilibrium is not reached from the prior.
Reading: once the readout equation is switched off, the theory's read-in + hidden equations have no state near the
theory's solution for a residual as large as the networks' population residual (He3 component 1.9x the theory's).
Not separated: a true absence of a solution versus a near-singular Newton system at the frozen-t saddle. Consistent
with hidden_equation_test.py ((E2) fed the networks' population t blows up). The residual the hidden neurons feel in
the exact posterior is the training residual (about half the population one), not the population residual used here.

## Full correction in the theory's own solver: c in the action and the readout, readout factor r (`perp_surrogate.py --r`)
Networks: hidden-neuron action term −1/2 g~ (t.e + c)^2 and readout r (sigma_A^2/N2)(t.e + c), r = 0.66 at P = 48000.
With c = gamma . e this is exactly the theory's own variational problem with sigma_A^2 -> r sigma_A^2, chi -> chi/r
(g~ unchanged) and target y + lambda gamma, so the original solver runs unchanged. P = 48000:

| | L3 | far He1 /net | large h3 /net | sd h1 | sd h3 | outer read-in |
|---|---|---|---|---|---|---|
| theory (r 1, c 0) | 0.947 | 11.6 | 0 | 0.218 | 0.063 | 150 |
| r 1, c on He1, He3 | 0.923 | 0.66 | 0 | 0.170 | 0.060 | 145 |
| r 1, c on all six modes | 0.925 | 0.61 | 0 | 0.176 | 0.059 | 157 |
| r 0.66, c 0 | 0.951 | 199 | 3.2 | 0.613 | 0.121 | 214 |
| r 0.66, c on He1, He3 | 0.927 | 173 | 1.1 | 0.583 | 0.116 | 208 |
| r 0.66, c on all six modes | 0.928 | 47 | 0.15 | 0.269 | 0.089 | 198 |
| networks | 0.899 | 2.0 | 17 | 0.164 | 0.093 | 57 |

- r < 1 makes the theory's hidden layer break symmetry much more, but along He1 (far-He1 group 11.6 -> 199 per
  network), not along He3.
- Adding c on all six modes pulls most of that back (199 -> 47) and brings the h3 width to the networks' (0.089 vs
  0.093), but the far-He1 group stays 20x too large, the large-h3 group stays absent (0.15 vs 17) and the outer
  read-in count goes the wrong way (198 vs 57). L3 moves only by the target shift.
- So with the measured c and r the theory still puts its extra hidden-layer structure along He1. Remaining
  difference found earlier in the action: G's He1-He3 correlation (−0.79 theory, −0.73 networks), which charges
  large-h3 states (h1 and h3 of the same sign) ~95 nats more.

## The read-in kernel decides whether the large-h3 group forms (`he13_test.py`)
Hidden layer and readout solved with the read-in kernel held fixed (J, qbar, M), P = 48000. Check: the theory's own
kernel, no correction, reproduces the theory (far He1 11.2, large h3 0, sd h1 0.215, sd h3 0.063).

| read-in kernel | correction | L3 | far He1 /net | large h3 /net | sd h1 | sd h3 |
|---|---|---|---|---|---|---|
| theory | none | 0.947 | 11.2 | 0 | 0.215 | 0.063 |
| theory | r 0.66, c | 0.923 | 44 | 0.07 | 0.272 | 0.078 |
| networks | none | 0.936 | 28 | 24 | 0.312 | 0.106 |
| networks | r 0.66, c | 0.914 | 77 | 100 | 0.440 | 0.136 |
| networks (sampled) | | 0.899 | 2.0 | 17 | 0.164 | 0.093 |

- With the networks' read-in kernel and nothing else changed, the theory's hidden layer forms the large-h3 group
  (24 per network against the networks' 17); with the theory's kernel it never does, with or without r and c.
  Caveat: that row's t was still moving 8 % per proposal round (the solve with a foreign kernel is not fully
  self-consistent, as in empirical_kernel_test.py).
- Adding r and c on top overshoots (100 large-h3, 77 far-He1): r and c were measured in networks whose read-in kernel
  is already the networks', so they are not additive with it.
- Setting only the He1-He3 correlation of one kernel to the other's value gives a matrix that is not positive definite
  (G is nearly singular), so the correlation alone could not be isolated this way.

## The theory's read-in distribution held at the sampler's outer count (`readin_at_count.py`)
Read-in prior tilted by exp(mu 1[|w0| > cut]) until the theory's outer count equals the sampler runs' mean; its w0
distribution against the sampler's (all post-burn snapshots) -> `plots/onset_checks/readin_at_count_d150.png`.
Outer well = cut < |w0| <= 1.2, far well = |w0| > 1.2 (neurons per network); sampler statistics from raw samples.

| P | | outer n | outer mean \|w0\| | outer sd | far well | inner sd | tilt mu (nats/neuron) |
|---|---|---|---|---|---|---|---|
| 12000 | theory, free | 77.6 | 0.734 | 0.025 | 0.56 | 0.127 | |
| | theory, held | 69.9 | 0.744 | 0.024 | 0.57 | 0.125 | −11 |
| | sampler | 70.0 | 0.740 | 0.057 | 0.00 | 0.119 | |
| 24000 | theory, free | 108 | 0.672 | 0.023 | 1.40 | 0.136 | |
| | theory, held | 50.7 | 0.727 | 0.016 | 1.23 | 0.119 | −127 |
| | sampler | 49.9 | 0.690 | 0.032 | 0.00 | 0.119 | |
| 48000 | theory, free | 150 | 0.612 | 0.022 | 2.63 | 0.144 | |
| | theory, held | 57.2 | 0.666 | 0.015 | 1.94 | 0.122 | −168 |
| | sampler | 57.0 | 0.752 | 0.026 | 0.33 | 0.143 | |

- P = 12000: at the same count the wells sit in the same place (outer 0.744 vs 0.740, inner widths 0.125 vs 0.119);
  the differences are the shelf between the wells (known: training-set directions), an outer well twice as wide in
  the sampler, and the far well (theory 0.56 per network, sampler none).
- P >= 24000: holding the count does not make the rest agree. The theory's outer well is in the wrong place at the
  sampler's count, and not by a common shift: too far out at 24000 (0.727 vs 0.690), too far in at 48000 (0.666 vs
  0.752). The theory keeps 1.2-1.9 far-well neurons per network; the sampler has 0-0.33.
- The tilt needed is −127 and −168 nats per outer neuron at 24000 and 48000 (−11 at 12000): in the theory the
  sampler's count is very far from where the read-in equation wants it.
- Not checked: whether the outer neurons' w0 equilibrate within the outer well in the sampler (their perpendicular
  directions do, within 50 iterations). If they do, the well position at a given count is a fair comparison. The
  far-well count is frozen in the sampler too.

## Effective ridge: the theory at P / f (2026-11-12)
The theory's readout uses the bare ridge, (sigma_A^2 K + kappa/P) t = y, and one residual for both the hidden-layer
term and the output. Fitting on a finite training set acts like a larger ridge kappa_eff = f kappa (smaller output,
smaller residual seen by the hidden layer). The theory depends on kappa and P only through kappa/P, so the theory with
kappa_eff at P is the theory at P / f (cached solutions, no new solves). f chosen per P so that the theory's L3 equals
the sampler's:

| sampler P | f | | outer n | outer mean \|w0\| | far well | inner sd |
|---|---|---|---|---|---|---|
| 12000 | 1.70 | theory at P | 77.6 | 0.734 | 0.56 | 0.127 |
| | | theory at P/f | 55.3 | 0.806 | 0.00 | 0.116 |
| | | sampler | 70.0 | 0.740 | 0.00 | 0.119 |
| 24000 | 2.25 | theory at P | 108 | 0.672 | 1.40 | 0.136 |
| | | theory at P/f | 72.6 | 0.748 | 0.40 | 0.125 |
| | | sampler | 49.9 | 0.690 | 0.00 | 0.119 |
| 48000 | 2.44 | theory at P | 150 | 0.612 | 2.63 | 0.144 |
| | | theory at P/f | 98.9 | 0.691 | 1.18 | 0.134 |
| | | sampler | 57.0 | 0.752 | 0.33 | 0.143 |

- The f that matches L3 (1.7-2.4) agrees with the measured readout factor (1/r = 1.5-2.4).
- At P = 48000 the read-in distribution moves most of the way toward the sampler's: outer count 150 -> 99 (57),
  outer well 0.612 -> 0.691 (0.752), far well 2.63 -> 1.18 (0.33), inner width matches. At 24000 the count and far
  well move toward the sampler but the outer well overshoots (0.748 vs 0.690). At 12000, where the theory already
  matched, it makes the count and outer well worse (55 vs 70, 0.806 vs 0.740).
- f was fitted to L3 here, one number per P. A real correction computes kappa_eff from the theory's own output-kernel
  spectrum (the perpendicular modes the readout can fit on P points).

## The networks' projections on their own training set (`train_projections.py`)
Same 9 networks per P as measure_perp_pull.py. On the training set, a_bar = (sigma_A^2/N2) <erf o h_k | t>_train with
t = (y - f_bar)/kappa exactly (slope 1.0000, R^2 1.00000): the theory's readout relation holds exactly when the overlap
is taken over the training set. The factor r = 0.66-0.74 came only from using population overlaps.

| | P = 24000 train | pop | theory | P = 48000 train | pop | theory |
|---|---|---|---|---|---|---|
| L3 | 0.938 | 0.839 | 0.913 | 0.966 | 0.897 | 0.947 |
| f_bar He5 | 0.083 | 0.172 | 0.128 | 0.060 | 0.123 | 0.097 |
| residual rms, x0 part | 0.102 | 0.216 | | 0.075 | 0.147 | |
| residual rms, perpendicular part | 0.095 | 0.155 | 0 | 0.069 | 0.098 | 0 |

- On the training set the networks learn He3 better than the theory says (L3 0.938 / 0.966 against 0.913 / 0.947);
  in the population worse. The training residual is about half the population one in rms (MSE ratio ~3).
- On the training set the residual's perpendicular part is as large as its x0 part.
- Per hidden neuron, training overlap against population: x0 part 0.65-0.74 of s (R^2 0.96-0.98), perpendicular
  part 0.37-0.53 of c (R^2 0.91).
- Hidden-neuron action term −1/2 g~ <erf o h | t>^2, group minus ordinary neurons:

| | training set | training, x0 part only | population (s + c) | population, s only |
|---|---|---|---|---|
| P = 48000, large h3 | −362 | −51 | −744 | −87 |
| P = 48000, far He1 | −35 | 0 | +9 | +49 |
| P = 24000, far He1 | −56 | −34 | −184 | −126 |

  With the training-set term the networks' totals (prior on h, He1-He3 block, + q_h prior + this term) are +24 for
  the large-h3 group at 48000 (occupancy 17 of 1481: ~4.5 nats) and +24 / +27 for far He1 at 24000 / 48000
  (occupancy ~7 nats), roughly consistent; with population overlaps they were −343 and +70. The perpendicular part
  still supplies most of the large-h3 term (−311 of −362).
- The standard effective ridge does not give it. For ridge regression on fixed features, kappa_eff = kappa /
  (1 - n_eff/P), n_eff the readout's degrees of freedom. Measured on the networks' own features on their training sets
  (n_eff = tr[Phi^T Phi (Phi^T Phi + kappa N2/sigma_A^2)^-1], 9 networks per P): n_eff = 123, 231, 321, 456 at
  P = 3000, 12000, 24000, 48000, so kappa_eff/kappa = 1.04, 1.02, 1.01, 1.01. Needed: 1.7-2.4 (fitted to L3) or
  1.74-1.90 (from the train/population residual ratio). The theory's own output kernel cannot give more than the
  networks' features do. The factor ~2 comes from the read-in and hidden weights fitting the training set, not from
  the readout.
- The project's kappa_eff (`lib/kappa_eff_solver.py` + `julia_lib/self_consistent_kappa_solver.jl`:
  kappa_eff = kappa + (kappa_eff/P) sum_k lam_k/(lam_k + kappa_eff/P)) is the same equation; it uses the arcsin
  (first-layer) kernel at unit scale, eigenvalues from 5000 sampled points. That is not converged in the sample count
  (P = 48000: 1.04, 1.11, 1.23, 1.37 from 2000, 5000, 10000, 15000 points). Converged (spectrum in degree shells,
  degree l: ~d^l/l! modes of eigenvalue c_l l!/d^l): kappa_eff/kappa = 3.97, 3.80, 3.75, 3.69 at P = 3000, 12000,
  24000, 48000, almost flat in P (dominated by the degree-3 shell acting as noise). The theory at P/f then gives L3
  0.295 / 0.629 / 0.767 / 0.863 (sampler 0.554 / 0.782 / 0.839 / 0.899) and at P = 48000 outer count 81, outer well
  0.727, far well 0.67 (sampler 57, 0.752, 0.33; bare theory 150, 0.612, 2.63); at 12000 and 24000 the read-in moves
  away from the sampler. For this model the kernel in that equation should be sigma_A^2 x the output kernel
  (sigma_A^2 = 8/1500), which gives 1.01-1.04 (above); the unit-scale arcsin kernel corresponds to a readout prior
  ~190x larger.

## Fixed theory, more data: the read-in wells move away from the theory's (2026-11-12)
Same runs as "Fixed theory, more data" (kappa ∝ P). Outer well = |w0| beyond the theory's barrier between the wells.

| theory point | | outer-well mean \|w0\| | outer-well sd | outer count |
|---|---|---|---|---|
| P = 3000 (dataset 2) | theory | 1.010 | 0.033 | 28 |
| | P x1 / x2 / x4 | 0.984 / 0.933 / 0.931 | 0.073 / 0.053 / 0.046 | 26-27 / 37 / 39 |
| P = 12000 (dataset 0) | theory | 0.734 | 0.025 | 78 |
| | P x1 / x2 / x4 | 0.735 / 0.702 / 0.687 | 0.056 / 0.039 / 0.035 | 72 / 96 / 114 |
| P = 24000 (dataset 0) | theory | 0.672 | 0.023 | 108 |
| | P x1 / x2 / x4 | 0.692 / 0.750 / 0.759 | 0.028 / 0.023 / 0.022 | 47 / 55 / 60 |

- More data empties the shelf between the wells and narrows the outer well toward the theory's width, but the outer
  well's position moves away from the theory's: the difference grows from 0.00-0.03 at P x1 to 0.05-0.09 at P x4.
- The direction follows the frozen count: where the count overshoots the theory's (theory points 3000, 12000) the
  well moves in; where it stays below (24000) the well moves out. More outer neurons, well further in, as in the
  theory held at a given count. So with the count frozen, more data does not bring the read-in distribution to the
  theory's; the well position is set by a count the sampler cannot equilibrate.

## Free energy against the outer count from the insert/delete move: not reliable as is (2026-11-12)
omega = -logA of each attempted move; inserts at n give <exp(-omega)> = p(n+1)/p(n), deletes the reverse; Bennett's
estimate of ln p(n+1)/p(n) from the existing P = 3000 runs (count.csv), where the count is known (~26-28, theory 28,
so the answer should be ~0):
- K = 40 (two chains, ~70 moves each way): −5.0 and −5.6; omega_insert + omega_delete ~ 52-55 nats of dissipation.
- K = 160 / 400 (4-11 moves each way): −0.6 / −1.0; dissipation ~40 / ~31 nats.
The estimate depends on the protocol length: with 30-55 nats dissipated the forward and reverse work distributions do
not overlap and the estimate is biased. A trustworthy value needs a few nats of dissipation (K in the thousands);
at P = 24000 one K = 400 move would take ~15 min.

## The shelf between the read-in wells is a few stuck neurons, not an equilibrium density (d = 150, P = 3000)
Six runs with snapshots (ray d150 r11-r13, onset P3000 r10-r12), 140 post-burn snapshots each (every 5 iterations,
700 iterations). Barrier region 0.3 < |w0| < 0.85: 2.9 neurons per snapshot.
- About half of that mass is 10 "resident" neurons (1-3 per run) that are in the barrier region in >80% of the
  snapshots. Each stays near its own position for the whole run (time sd of |w0| 0.07, median; the region is 0.55
  wide), and the positions are scattered: 0.39, 0.42, 0.43, 0.48, 0.49, 0.53, 0.57, 0.65, 0.68, 0.68.
- The other half is 128 brief visitors (<20 % of snapshots each, time sd 0.013): excursions from the wells' edges.
- So the flat shelf in the pooled histogram is the sum of ~10 neurons parked at scattered places, not a sampled
  density across the barrier: its shape depends on where those neurons happened to stop. The equilibrium mass there is
  not measured by these runs.
- Their trajectories (9 distinct neurons; ray r11 and onset r11 are the same trajectories), plotted in row 6 of
  `plots/ray/readin_ray_fcn2_vs_fcn3.png`: all have q / q_prior 1.2-1.5, above both wells (inner 1.0, outer 0.46).
  Two of them start in the outer well (|w0| ~ 0.9-1.0, q / q_prior ~ 0.4-0.5) and move into the barrier region
  within ~100 iterations as q grows to ~1.1-1.35: outer neurons leaving the outer well through a high-q path. The
  rest wander (time sd 0.05-0.08) with slow drifts, mostly inward (−0.02 to −0.34 per 1000 iterations; one +0.2).
  At that pace crossing the 0.55-wide region takes thousands of iterations; the runs have 700 after burn-in.
- Reading: these are neurons in slow transit through a high-q channel between the wells (the path from the outer well
  to the inner one goes through large q), caught mid-way, not parked at fixed points. The flat shelf is their pooled
  snapshot. Equilibrium needs runs long enough for them to finish crossing, or a move that changes w0 and q together.

## Mass between the read-in wells at the fixed-theory point P = 12000: wider outer wells, not barrier crossing
Runs: dataset 0 at P = 12000, 24000, 48000 (kappa ∝ P; theory fixed at the P = 12000 point).
- Between the wells (0.3 < |w0| < 0.62): 2.7-4.1 neurons per snapshot, all brief visitors (45-118 distinct neurons,
  median stay 5 iterations, longest 40-95), none resident. Visitors have the outer well's q at x2 and x4 (0.58-0.61).
  No alignment with the training-residual direction at x4 (|cos| 0.055, random 0.065).
- Counts per |w0| bin against the theory (same number of neuron-snapshots), 0.30-0.36 / ... / 0.54-0.62:
  x1 53, 72, 58, 101, 274; x2 2, 8, 3, 18, 343; x4 1, 1, 3, 20, 552; theory 0.14, 3e-5, 4e-8, 2e-7, 0.04.
  Most of the excess is on the inner flank of the outer well.
- Each outer neuron's own |w0| fluctuates more than the theory's well: sd over time 0.042 / 0.035 / 0.030 at
  x1 / x2 / x4 (theory 0.025). Not heterogeneity (per-neuron means spread 0.004-0.010) and not a common motion of the
  outer neurons (sd of their snapshot average 0.004-0.005): individual, and narrowing toward the theory with more data.
- Read-in cavity test (`readin_cavity.py`, P = 12000 dataset 0, last saved network): one outer neuron's w0 moved
  inward, everything else fixed, exact energy with the readout integrated out. From the outer-well bottom to |w0| =
  0.6 it matches the theory's -log p(w0) to ~1 nat (neuron 843: −3.1, +0.9, +11.6 vs theory −3.0, +1.0, +11.2), so
  with the environment frozen the well has the theory's width. Further in it keeps rising (relaxing the neuron's
  outgoing weights and q, three neurons: 67-84 nats at |w0| = 0.3 and 75-106 at the inner well, where the theory is
  2-5 nats below the outer well; the energy rises monotonically, no second well): a single neuron cannot
  leave the outer well without the rest of the network rearranging.
- Reading: the excess mass between the wells here is the tail of outer wells that are 20-70 % wider per neuron than
  the theory's, a finite-P effect (it narrows toward the theory as P grows at fixed theory), not an unequilibrated
  barrier population. Crossing the barrier needs a collective rearrangement, consistent with the frozen count.

## Continuing a d = 150 run for 3000 iterations: the shelf drains slowly, two neurons stay (2026-11-12)
Run `runs/*_shelf_d150_P3000_*_r113`: started from the final state of ray d150 r13 (`--init-from`), same physics,
3000 iterations, snapshots every 5. Band 0.3 < |w0| < 0.85.
- Neurons between the wells per snapshot: 4.45 (original run) -> 3.94 (iterations 0-500) -> 3.06 (500-1500) -> 2.76
  (1500-3000). Outer count (|w0| > 0.85) 26.1 -> 25.8 -> 25.5 -> 25.2. One full inner/outer crossing (neuron 1434).
- The three resident neurons, against the drift extrapolation (expected exits ~280, ~1600 and ~210 iterations):
  1493 left inward after ~500 iterations, to |w0| ~ 0.20 with q / q_prior 1.77 (not the inner well's ~1.0);
  252 stayed at |w0| 0.46-0.52 for all 3000 iterations (q / q_prior 1.43); 370 stayed at 0.39-0.40 (q / q_prior
  1.61). The extrapolation was wrong for two of three.
- So part of the shelf is slow transit that does drain (about 40 % over 3000 iterations), and part is neurons that sit
  at intermediate w0 with large q (1.4-1.8 x prior) for at least 3700 iterations: a long-lived high-q state that the
  runs do not resolve as transient. Whether the theory's (w0, q) action has a local basin there is not checked.
