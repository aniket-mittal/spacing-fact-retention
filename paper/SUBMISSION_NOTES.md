# Submission notes: spacing paper

## Target venues
- **Primary: TMLR** — archival, judged on correctness + interest, no deadline. A clean, replicated
  positive finding with a confound-controlled design fits well.
- **Parallel: a NeurIPS/ICLR workshop** on science-of-deep-learning, data-centric ML, or efficient
  training. Non-archival, so no conflict with TMLR.

## TMLR author note (paste into the submission comment field)

TMLR reviewers judge whether the claims are supported by accurate evidence and whether some audience
would find the work interesting.

**Claims and support.** (1) Holding exposure count and total training fixed and varying only the
spacing of a fact's repeated exposures, in-weight fact recall is a non-monotone (inverted-U)
function of spacing, with an interior optimum. The effect is large (about 0.7 in recall between the
optimum and the massed/maximally-spaced endpoints) and stable across three seeds
(Section~4, Figure~1). (2) The optimum governs durable retention, not just end-of-training recall:
after continued filler-only training, well-spaced facts survive better, and the fraction of recall
retained rises with spacing (Section~5, Figure~3). (3) The inverted-U replicates across four model
families; its magnitude tracks how hard the fact is to memorize, shrinking toward the ceiling for
models that memorize easily (Sections~6--7, Figure~2). (4) A confound control (equalizing
memorization across learning rates by scaling epochs inversely with the learning rate) shows the
optimum does not move systematically with learning rate; an earlier uncontrolled sweep that appeared
to show a shift is explained as a failure-to-memorize artifact, which we report explicitly.

**Positioning.** The result reconciles a reported null (Tirumala et al., 2022: spacing has minimal
effect on memorization) with the classical human-memory spacing ridgeline (Cepeda et al., 2008: an
inverted-U in the interstudy interval). We measure semantic fact retention across the full spacing
range and recover the ridgeline in weight space; the prior null is consistent with sampling a flat
region of the same curve.

**Scope and honesty.** We use LoRA on synthetic facts with invented entities to isolate the
manipulation. We do not claim a learning-rate scaling law for the optimum; that claim was tested and
did not survive the confound control, and we say so.

**Audience.** Anyone teaching facts to a model by repeated exposure (editing, personalization,
continual learning) gets a concrete, free schedule knob; researchers on memorization dynamics and
data ordering get an in-weight analogue of the spacing ridgeline and a reconciliation of prior
disagreement.

**Reproducibility.** All experiments run on a released Modal + PEFT harness (exp_spacing_law.py),
covering the gap sweep, the family replication, the retention measurement, and the
confound-controlled learning-rate sweep.

## Files
- spacing.tex (paper), fig_spacing_invertedU.pdf, fig_spacing_families.pdf, fig_spacing_retention.pdf
- make_spacing_fig.py, make_spacing_families_fig.py (regenerate figures)

## Remaining optional polish
1. Expand the fact pool / add a second neutral filler distribution for the headline curve to widen
   the population estimate.
2. Vary the length of continued training in the retention experiment to test whether the optimum
   shifts with retention interval, as it does for human learners (Cepeda et al., 2008).
