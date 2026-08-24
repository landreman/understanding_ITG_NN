# Claude summary

## What the step was for

Your trained model is an *ensemble*: 100 separate neural networks, each trained on the same data but with different architectures (different layer widths, etc.), whose predictions get averaged. Earlier steps (S04, S05, S08) had dissected individual networks — mostly the single best one — and found internal quantities that seem to correspond to physical concepts. That raises an obvious question: **are those findings idiosyncrasies of one network, or do all 100 networks converge on the same internal computations?** If independent-ish networks all invent the same internal quantity, that quantity is much more likely to reflect real structure in the physics rather than an accident of training. Step 10's goal was to test that, by comparing the networks' internals to each other.

The key comparison point is each network's **bottleneck**: a narrow layer in the middle of the network where the 96-point geometry profiles have been compressed down to a handful of numbers (between roughly 8 and 30, depending on the member). Each of those numbers is a "unit" — a learned summary statistic of the geometry. The bottleneck is where networks of different shapes can be compared unit-by-unit, so the central task was to match up units across networks: "unit 3 of network A computes the same thing as unit 7 of network B."

## How the matching worked, briefly

Two bottleneck units in different networks were declared a match only if they agreed in several independent ways at once, on the same frozen panel of 1,000 rows (one flux tube from each of 1,000 equilibria):

1. **Their activations correlate** — they output similar values across the same inputs.
2. **They still correlate after removing the easy explanation.** Any two units that both track the final heat-flux answer will correlate trivially. So the analysis subtracted out the part of each unit explained by the flux and the two gradient drives, and required the *leftover* signals to still match. This guards against "label-only" matches.
3. **The match is stable under resampling.** The 1,000 equilibria were resampled 100 times (a *bootstrap* — redrawing the data with replacement to see whether a result depends on which particular examples happened to be included), and the pairing had to reappear in at least 70% of resamples.
4. **They matter to their networks in the same way.** Each unit was *ablated* — its value forcibly replaced by its average, and the change in the prediction recorded for every row. Two matched units had to shift their networks' predictions in the same direction row-by-row. Crucially, this agreement was required separately on the 240 stable/near-floor rows and the 760 unstable rows, so opposite behavior in one regime couldn't hide inside a good average.

Groups of mutually matched units across many networks are called **motifs** — candidate "shared computations."

Two supporting analyses ran alongside. **CKA** (Centered Kernel Alignment) is a single number, 0 to 1, measuring whether two networks' internal representations have the same *geometry* — whether inputs that look similar to one network also look similar to the other — even when the layers have different widths. And a **clustering** analysis grouped all 100 members by their predictions, input sensitivities, ablation behavior, and concept profiles, to see whether any subpopulations (e.g. the four narrowest-bottleneck members, or low-ranked members) behave qualitatively differently.

## What was found

**The strict causal evidence for shared computations is much thinner than surface similarity suggests.** Matching started with 582 above-threshold unit pairs among the top-10 networks. After all the gates — especially the requirement that ablation effects agree separately in both output regimes — only **163 pairs survived**, and a post-run audit was what forced that regime-separated check: 334 of the preliminary pairs looked causally similar on the four-number summary but actually had *opposing* effects in at least one regime (some had cosine similarity near −0.8 on stable rows, i.e. nearly opposite). This is exactly the failure mode the project's "keep signed, per-regime results" rule exists to catch.

**Eight motifs emerged, five of them substantial** — appearing in 9, 7, 7, 6, and 4 of the top 10 networks. So there *are* internal quantities that most of the best networks compute in a functionally and causally consistent way. But the report is careful about two caveats:

- The count "eight" depends on a similarity threshold: sweeping it from 0.50 to 0.80 gives 14, 12, 8, or 4 motifs. Eight is a description at one chosen cutoff, not a natural number of shared mechanisms.
- **Only one motif has a physical name.** The seven-member `motif_001` contains a unit that step S05 had linked to the circular mean of the $f_Q$ integrand from your paper — but that anchor unit lives in an atypical member (the narrowest bottleneck, an outlier in the clustering), so it's a tentative hint, not an identification of all seven. The other seven motifs are either "screened by S05 but no supported name found" (motifs 002–004) or "never screened" (005–008). Functional matching tells you units *behave alike*, not what they *mean*.

**All 100 networks are representationally very similar — but that similarity is misleadingly generous.** Median CKA is 0.948 at the first internal layer, declining to 0.814 at the bottleneck. High CKA everywhere, including between the top-ranked and lowest-ranked members, and between narrow- and wide-bottleneck members. Yet the four narrow-bottleneck members are clearly *different* on the combined evidence distance (median distance 3.15 to wide members versus 1.18 wide-to-wide). The report's own conclusion: representation-geometry similarity alone would have overstated how mechanistically alike the networks are. CKA was correctly treated as supporting evidence only.

**Validation rank doesn't organize anything.** Better-performing members are not more "central" or more similar to each other (rank-vs-distance correlation 0.118, not significant; lower-ranked cohorts have equally high CKA). Whatever separates a rank-5 member from a rank-90 member, it isn't a different internal representation in any way this analysis can see.

**One important honesty note the report flags:** all 100 members were trained on the same data with related architectures. So when seven networks share a motif, that recurrence has an unmeasured "shared training" floor — it is not seven independent discoveries of the same physics, the way seven independent research groups finding the same result would be.

## The conclusions in one paragraph

The ensemble's networks agree broadly on *how they represent* the geometry
inputs, and a small core of genuinely shared, causally consistent internal
computations exists — eight motifs at the registered threshold, five spanning
most of the top networks. But the step's most valuable outcome is arguably
negative: naive similarity measures (activation correlation, CKA, pooled causal
summaries) would have suggested far more agreement than survives strict,
regime-separated causal testing (163 of 582 candidate pairs), and only one motif
so far connects to a named physical quantity, tentatively. The shared motifs are
now cataloged (in
[motif_catalog.csv](reports/xai/S10_artifacts/motif_catalog.csv)) as the raw
material for later steps — S11, S12, and the eventual synthesis in S14 — whose
job includes putting physical names on the seven motifs that don't have one yet.

-------------


# S10 Codex summary — what the networks share

## What this step asked

The ensemble contains 100 separately trained neural networks. They make very
similar heat-flux predictions, but that does not guarantee that they learned the
same geometric reasoning. S10 asked whether their internal representations and
their important bottleneck units correspond across networks, and whether any
shared pattern persists outside the ten networks with the best stored validation
scores.

The analysis explains each network's native prediction
$\max(\log Q,-2)$. It uses the exactly shift-invariant version established in
S02, so moving the common origin around the periodic flux tube cannot change the
object being compared. The 1,000-row S01 panel contains one tube from each of
1,000 equilibria: 240 rows where the model is at or near the clipped output floor
and 760 unstable rows above it.

## What “the same internal feature” means here

A bottleneck unit is one of the 7–32 numbers through which all geometric
information must pass before the network makes its prediction. Two units were
matched only when several kinds of evidence agreed:

- they varied similarly over the same equilibria;
- they still varied similarly after removing simple associations with heat flux
  and the two gradient inputs;
- they had similar relationships to named geometry concepts and spatial scales;
  and
- replacing each unit by its typical value changed the two networks' predictions
  similarly.

That replacement is a diagnostic edit inside a network. It is not a physically
realizable change to a stellarator equilibrium, so it explains the networks and
does not by itself establish plasma causality.

The matching was repeated while resampling whole equilibria. Because this panel
contains exactly one tube per equilibrium, that is numerically the same as row
resampling here, while preserving the correct rule for future multi-tube cohorts.
It estimates how often a match would recur if the panel contained a different
sample of equilibria. Units were allowed to remain unmatched; the calculation did not force
every internal number into a shared story.

## Main result: shared motifs exist, but far fewer than a simple comparison suggests

The first comparison found 582 plausible unit pairs among the top ten networks.
497 looked stable under equilibrium resampling and under a compact summary of
their effects. A stricter audit then found a serious problem: that compact
summary could look similar even when the signed effects opposed one another
within the stable or unstable regime.

After requiring agreement separately in both regimes, only **163 pairs** remained.
For the motif catalog, a separate stricter threshold retains 74 of
those edges; four inconsistent unions are then removed to keep at most one unit
from each network in a motif. The remaining 70 edges form **eight shared motifs**.
Five motifs occur in at least four of the top ten networks; the largest occurs in
nine. The correction rejected 334 preliminary edges and is scientifically
important: averaging across regimes would have produced 33 apparently shared
motifs instead of eight.

The motif count depends on the explicit 0.70 catalog threshold: using
0.50/0.60/0.70/0.80 gives 14/12/8/4 motifs. The 0.70 value was present in the
first S10 code before the regime audit and matches the earlier pooled-effect
gate, but eight should still be read as a thresholded catalog count rather than
a unique natural number.

The surviving pairs agree in both direction and useful scale. The larger of the
two root-mean-square prediction effects is typically **1.33 times** the smaller;
90% of pairs are within **2.17 times**, and the maximum ratio is **4.05**. Thus
the cosine direction test is backed by an explicit magnitude comparison. The
same comparison remains similar when kept separate by regime: stable/near-floor
median/p90/max **1.41/2.50/5.04**, unstable **1.34/2.21/3.98**.

Only one of the eight motifs contains a unit that S05 had given a supported
physical name. It includes the leading network's unit associated with a
parallel-window average of the paper's $f_Q$ integrand, and corresponding units
occur in six other networks. But those six units have not independently earned
that name. It is a promising anchor, not a seven-network identification. The
anchoring unit also belongs to the narrowest network and is an outlier under the
average-linkage description, so it is not representative of the ensemble. S05 screened units from only the
first four motifs; motifs 005–008 contain no unit S05 examined. “Unresolved” for
those four therefore means not yet screened, not a demonstrated vocabulary gap.

## The networks look similar in broad outline

Centered Kernel Alignment, or CKA, compares the geometry of two internal
representation spaces even when they contain different numbers of units. A CKA
score near one means that examples have similar relative arrangements in the two
spaces; it does not prove that the networks perform the same computation.

Across all 4,950 pairs of networks, median CKA decreases from **0.948** in the
first spatial layer to **0.814** at the final invariant bottleneck. This shows
broadly similar representation geometry at every measured layer. It does not
show that networks become more individual with depth because this run did not
register a permutation or chance baseline for comparing layers. Removing the 5%
most extreme probe examples changes the typical pair score by a median of
0.006–0.022, so the
raw scores are not usually driven by a few outliers.

The uncertainty ranges for CKA use only 20 resampled panels. They are a coarse
sensitivity check, not a precise confidence interval. The exact all-pair scores
are complete; a planned 100-resample version exceeded the step's computation
budget and is recorded as deferred.

## Validation rank is not the organizing principle

Average linkage places 95 of the 100 networks in one main cluster when
predictions, scaled input sensitivities, bottleneck interventions, and concept
profiles are combined. Other standard linkage choices produce 82/12/5/1
clusters with different member identities, so the cut does not establish five
robust outliers. Distance from the most central member has only a
weak relationship with stored validation rank (rank correlation 0.118, with
$p=0.243$). Networks ranked 51–100 are no less similar at the bottleneck than
the top ten: their median within-cohort CKA is 0.816, compared with 0.796 for the
top ten.

The four narrow networks, with at most 11 bottleneck units, are more unusual in
the combined evidence. Their median distance to wide networks is 3.153, compared
with 1.177 between two wide networks, and three of the four sit outside the main
cluster under average linkage. Because that membership changes with linkage,
the narrow-member conclusion rests on the continuous 3.153 versus 1.177
distance comparison. Their bottleneck CKA with wide networks is almost unchanged
(0.813 versus 0.814 between wide networks). The narrow networks do not simply
lose the common representational scaffold; they differ in how they use it.

## What to carry forward

The strongest conclusion is methodological and scientific at once: shared
activation patterns are abundant, but shared signed effects in both physical
regimes are much rarer. S11 and S12 should use the eight strict motifs, retain
the regime-specific signs, and keep the other 334 preliminary correspondences as
negative evidence rather than silently recovering them through averaging.

All 100 networks share training data and an architecture family. The recurrence
of a motif can therefore reflect shared training bias as well as a genuinely
necessary internal feature; S10 does not measure that shared-training floor.

The seven unnamed motifs are also useful. For motifs 002–004, S05 screened at
least one unit without finding a supported name; motifs 005–008 have not yet
been screened by S05. Both are concrete targets for later disagreement analysis
and interpretable distillation, not reasons to attach a post-hoc name.
