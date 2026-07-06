# Spacing and Fact Retention

Code and paper for **"An Optimal Repetition Spacing for In-Weight Fact Retention."**

Repository: <https://github.com/aniket-mittal/spacing-fact-retention>

## Summary

When a language model is fine-tuned to memorize a fact from repeated exposures, does it matter
whether those exposures are packed together or spread across training? We show it matters a great
deal, and the relationship is non-monotone. Holding the number of exposures and the total amount of
training fixed, and varying only the spacing between a fact's repeated exposures, recall follows an
inverted-U: both cramming the repetitions together and spreading them maximally far apart give poor
retention, while an intermediate spacing is best. On a 360M-parameter model the optimum recovers
near-perfect recall (0.99) while the massed and maximally spaced schedules reach only 0.22 and 0.31.

The inverted-U replicates across four model families, its magnitude tracks how hard the fact is to
memorize, and it governs durable retention (well-spaced facts survive continued training better).
The result reconciles a reported null (spacing has little effect on memorization) with the classical
human-memory spacing ridgeline (an inverted-U in the interstudy interval) as two views of one curve.

## Repository layout

```
paper/
  spacing.tex                    paper source (TMLR format)
  tmlr.sty, tmlr.bst,            TMLR style files (so it compiles standalone)
  fancyhdr.sty, math_commands.tex
  fig_spacing_invertedU.pdf      Figure 1: the inverted-U (SmolLM2-360M)
  fig_spacing_families.pdf       Figure 2: replication across four families
  fig_spacing_retention.pdf      Figure 3: recall vs durable retained recall
  make_spacing_fig.py            regenerates the inverted-U figure
  make_spacing_families_fig.py   regenerates the families + retention figures
  SUBMISSION_NOTES.md            venue notes + TMLR author note
scripts/
  exp_spacing_law.py             the full harness: gap sweep, family replication,
                                 retention measurement, and confound-controlled LR sweep
```

## Reproducing

Experiments run on [Modal](https://modal.com) with Hugging Face Transformers + PEFT.

1. Install Modal (`pip install modal`) and authenticate (`modal token new`).
2. Set a Hugging Face token as an environment variable (`HUGGINGFACE_TOKEN`) for gated models.
3. Run the experiment modes:
   ```bash
   modal run scripts/exp_spacing_law.py                    # main gap sweep (inverted-U), 3 seeds
   modal run scripts/exp_spacing_law.py --families         # replication across 4 model families
   modal run scripts/exp_spacing_law.py --retain           # recall vs durable retained recall
   modal run scripts/exp_spacing_law.py --scaling          # confound-controlled learning-rate sweep
   modal run scripts/exp_spacing_law.py --validate         # quick smoke test
   ```

The design holds exposure count and total training fixed across all spacings, so only the temporal
placement of a fact's repeats changes. Results are written to a Modal volume and echoed to stdout.
Figures are regenerated with `python paper/make_spacing_fig.py` and
`python paper/make_spacing_families_fig.py`.

## Building the paper

The paper uses the TMLR style file (double-blind by default). To compile:

```bash
cd paper && pdflatex spacing.tex   # (run twice for references)
```

For the de-anonymized camera-ready or preprint, change `\usepackage{tmlr}` to
`\usepackage[accepted]{tmlr}` (or `[preprint]`) at the top of `spacing.tex`.

## Citation

If accepted, cite the TMLR version. A BibTeX entry will be added on publication.
