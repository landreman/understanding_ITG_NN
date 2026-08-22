# Medoid
--------

A medoid is a real example chosen to represent the “middle” of a collection of examples.

In this repository, each example contains a magnetic-geometry profile: seven geometry quantities measured at many positions along a magnetic field line. The medoid is the one observed geometry profile that is most typical of a reference collection.

A useful analogy is choosing a representative person from a group:

- The “average person” might be assembled by averaging everyone’s height, age, and other attributes. That artificial combination may describe nobody who actually exists.
- The medoid is the real person who is closest to that group average.

Here, the same distinction is important:

- A median or “robust constant” reference is constructed statistically and may not correspond to any observed geometry.
- The medoid is one complete geometry that actually occurs in the data.

### How this repository chooses it

The code first constructs a robust center:

1. At every field-line position and for every geometry channel, it takes the median across the reference geometries.
2. It measures how far each real geometry is from that median profile.
3. Because the seven channels have very different numerical scales, it scales their differences before comparing them. Otherwise, the channel with the largest numbers would dominate the choice.
4. The real geometry with the smallest overall scaled distance is selected as the medoid.

So “medoid” here means:

> The observed geometry profile closest to the point-by-point median geometry of the reference set, after accounting for the different scales of the seven channels.

It can also be calculated within a particular equilibrium class, although S06a uses a single medoid from the full background-support collection.

### Why it appears in the explanation results

The repository uses Integrated Gradients to explain a network prediction. You can think of this method as asking:

> As we move from a reference geometry to the geometry being explained, which input features are responsible for the change in the network’s prediction?

That reference geometry is called the baseline. “Medoid IG” means that the medoid is used as the starting geometry.

For an input \(X\) and medoid \(M\), Integrated Gradients follows an artificial path from \(M\) to \(X\). It divides the prediction difference

\[
f(X)-f(M)
\]

among the geometry channels and field-line positions. A positive attribution means that moving from the medoid toward the input at that feature contributes toward raising the predicted clipped log heat flux; a negative attribution contributes toward lowering it.

This is a comparison, not an absolute declaration of importance:

> Medoid IG explains why the network treats this geometry differently from the selected typical observed geometry.

Change the baseline, and the question—and potentially the answer—changes.

### What “observed” does and does not guarantee

The medoid is tagged as an `observed_comparison` because its endpoint is a real observed geometry. That is an advantage over a wholly synthetic reference.

But the intermediate geometries used by Integrated Gradients are numerical mixtures between the medoid and the input. They need not correspond to physically realizable magnetic equilibria. Thus medoid IG explains the network’s behavior along that constructed path; it does not prove that a real stellarator could be modified along that path or would produce the predicted plasma response.

Also, “typical” is limited to the reference collection. The medoid is not necessarily:

- the most common geometry in nature;
- physically optimal;
- representative of every equilibrium class;
- close to every input being explained;
- a geometry with average heat flux.

It is simply the most central observed profile according to this repository’s defined distance measure.

### How to read the reported medoid results

In S06a, medoid IG was one of several competing explanation methods:

- The early 64-row pilot selected medoid IG.
- The larger, corrected 128-row production analysis selected low-pass IG instead.
- Medoid IG remained a sensitivity check rather than the final primary method.
- The medoid and robust-constant attribution maps had rank correlation 0.749. That is fairly substantial agreement, but far from identical.
- The selected low-pass and robust-constant maps correlated only 0.432.

The main scientific lesson is not that the medoid was “wrong.” It is that the apparent location and importance of features depend meaningfully on what geometry is used as the comparison point. The repository therefore treats medoid results as a check on whether conclusions survive a reasonable alternative baseline.

The safest interpretation of a feature is:

> Relative to a typical real geometry from the reference collection, this feature helped the network move its prediction up or down.

It would be too strong to say:

> This feature is inherently important to turbulence, independent of the comparison geometry.

The implementation is in
[perturbations.py](/Users/mattland/understanding_ITG_NN/understanding_ITG_NN/itg_nn/xai/perturbations.py:171),
and the baseline-sensitivity results are discussed in
[S06a_attribution_benchmark.md](/Users/mattland/understanding_ITG_NN/understanding_ITG_NN/reports/xai/S06a_attribution_benchmark.md:147).


# Head
------

The network for each ensemble member is really two stages bolted together:

1. **A cyclic-convolutional encoder** that takes the geometry input (channels describing the flux-tube shape) and a couple of scalar gradients ($a/L_T$, $a/L_n$), and squeezes them down into a small vector of numbers — the "invariant bottleneck," written $\bar u_m$. This is the part that does the heavy feature-extraction from raw geometry.

2. **The "head"** — a small function (in this codebase it's literally a `HeadFunction`, e.g. [bottleneck.py:43](itg_nn/xai/bottleneck.py:43)) that takes that compressed bottleneck vector plus the two gradient scalars and turns them into the network's single scalar prediction (the clipped `max(log Q, -2)` value). It's the last stage — the part that *combines* the compressed summary into a final answer, as opposed to the earlier stage that *extracts* the summary from geometry.

So "head" is standard ML jargon for "the final layer(s) of a network that map its internal representation to the output," by analogy to a body-with-a-head: the bulk of the network does the representation learning, and the head sits on top and produces the answer. It's a common term because the same encoder body is often reused with different heads for different tasks — though here there's just one head per ensemble member producing one output.

## Why S04 cares about it specifically

The whole point of [PLAN.md:617-654](PLAN.md:617) (step S04, "Anatomy of the
invariant bottleneck") is to open up that head and ask: given the compressed
bottleneck $\bar u_m$ a member computes, *how* does the head turn those few
numbers into the final heat-flux prediction? Is it basically linear, or does the
head do real nonlinear combining (interactions between bottleneck units,
curvature, etc.)? That's what the Shapley values, single/pairwise interventions,
and PDP/ICE curves in [bottleneck.py](itg_nn/xai/bottleneck.py) are probing —
they treat `head` as a black-box function and ask which of its 8–32 inputs
(bottleneck units + 2 gradients) drive the output and how they interact.

# À trous
---------

**"À trous"** (French for "with holes") is a technique for computing a convolutional network's pooling/conv chain *without* downsampling, while still reproducing exactly what the strided (downsampled) version computes. In this codebase it shows up in [symmetry.py:81](itg_nn/xai/symmetry.py:81), [PLAN.md:51](PLAN.md:51), and the S02 report.

**Why it matters here:** The trained network processes a 96-point geometry signal (values around a flux tube) through a chain of conv + pooling layers with stride > 1 (e.g. stride-2 max pools), each of which halves the resolution. That's efficient, but it throws away *where along the flux tube* each feature came from — by the time you reach the bottleneck, you only know the pooled/averaged value, not its position.

The "à trous" version is a reimplementation of that exact same pooling chain, but instead of shrinking the array at each layer, it keeps it at full 96-point resolution and instead **dilates** the convolution/pooling filters ("puts holes" between the taps) by the cumulative stride so far. Concretely, a stride-2 max-pool followed by a stride-2 conv becomes, in the à trous version, a same-length operation where the conv looks at every-other-sample instead of adjacent samples. This is mathematically equivalent to running the strided version at all possible shift-offsets ("phases") simultaneously and keeping them side by side instead of picking one and discarding the rest.

**The payoff:** averaging the à trous output over its position axis reproduces *exactly* what the real, downsampled network computes (verified to ~2e-6 per the S02 report) — but now you also get a full-resolution, position-resolved quantity (called `rho` in the code) that tells you which positions along the flux tube drove the prediction. That's the object later steps (05, 07) use to make claims like "this part of the flux tube mattered."

This is the same trick used in "atrous convolution" / dilated convolution in
image segmentation (e.g. DeepLab) — same math, different application.

# rho
-----

`\rho_{m,c}(z)` is the "equivariant density" (an XAI construct, invented in S02)

It's a made-up diagnostic object, not physics. Some background on the network:

- Each ensemble member takes a 7-channel geometry profile sampled at 96 points along the field line (`z` is position along that line, and it wraps around — hence "cyclic").
- It runs convolutions and max-pools down that axis, then averages over position ("global average pooling", GAP), producing one number per **unit** — a bottleneck vector `bar_u`. That vector plus the two gradient scalars `a/L_T`, `a/L_n` goes into a small dense head (MLP) that emits the prediction.

The averaging step throws away *where* along the field line each feature fired. `rho` recovers it. Implemented in [`equivariant_density`](itg_nn/xai/symmetry.py:135), it re-runs the same trained convolutions but **stride 1 with dilation** (the "à trous" trick) instead of stride-2 pooling, so you get an activation at all 96 positions instead of the 3 that survive the trained downsampling. So:

- `rho[m, c, z]` = how strongly unit `c` of member `m` responds to the geometry at position `z`.
- Averaging over `z` gives back exactly the bottleneck the trained model uses: `mean_z rho = bar_u` (verified in [S02_symmetry.md:156](reports/xai/S02_symmetry.md:156) to 1.9e-6).

Two jargon terms worth unpacking:

- **Equivariant**: shift the input along the field line and `rho` shifts by the same amount — `rho(S_k X) = S_k rho(X)`. That's what makes it a legitimate positional map rather than an artifact; a bug where the map got rolled sideways for even-kernel architectures was caught exactly by this test ([S02_symmetry.md:24](reports/xai/S02_symmetry.md:24)).
- **Invariant** (the contrasting term): `bar_u` and the prediction `tilde_f` are unchanged by such a shift, because averaging destroys the position information.

Calling it a "density" is by analogy: it's a non-negative (post-ReLU) quantity distributed over position whose total/mean is the meaningful summary, like a mass density integrating to a mass.

Two derived uses you'll see: **dead units** are defined as those whose `|rho|` never exceeds 1e-12 across the panel — units that contribute nothing ([S02_symmetry.md:213](reports/xai/S02_symmetry.md:213)); and S05/S07 regress `rho_{m,c}(z)` against physical concept traces to ask "what geometric feature does this unit detect, and where?"