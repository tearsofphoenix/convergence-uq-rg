# Paper 3 Contribution Summary

Date: 2026-04-02  
Author: Codex

## One-Sentence Positioning

`paper3_rg.tex` is now positioned as a benchmark-design negative-result paper:
it tests whether standard neural networks show RG-like advantages on a local
Ising block-spin task, and argues that strong claims require stricter protocol
design than is common in the literature.

## Primary Contribution

The main contribution is methodological rather than architectural:

- a Wolff-sampled, matched-budget same-scale benchmark
- explicit comparison against linear baselines
- clear separation between primary evidence, methodological clarification, and
  appendix-level diagnostics

## Supported Main Claim

On the 2D Ising majority-vote block-spin task:

- the task is learnable
- a linear baseline outperforms a standard fully connected MLP at criticality
- this remains true after replacing local Metropolis updates with Wolff cluster
  sampling

Therefore, the paper supports a scoped negative result:

> On majority-vote block-spin tasks dominated by a local linear-threshold rule,
> extra fully connected depth does not provide an RG-like advantage.

## What The Paper Does Not Claim

- It does not claim that neural networks cannot learn RG-like structure in
  general.
- It does not claim that whole-lattice transfer has been definitively falsified.
- It does not claim that Jacobian summaries already extract universal critical
  exponents.
- It does not claim that the XY appendix is a fully mature positive result.

## Evidence Tiers

### Primary evidence

- Wolff-sampled same-scale Ising benchmark

### Methodological clarification

- reduced-budget whole-lattice tiled transfer
- Wolff mixing diagnostics

### Diagnostic appendices

- XY nonlinear coarse-graining benchmark
- Jacobian batch summaries
- legacy pilot sanity checks
- turbulence note

## Why This Version Is Stronger

- Critical slowing down is no longer the dominant weakness of the main
  same-scale benchmark.
- The negative result is no longer “variance-driven ambiguity”; it survives
  under a stronger sampling protocol.
- XY and Jacobian branches now function as scope tests, showing where the main
  Ising conclusion should and should not be generalized.

## Suggested Submission Framing

- benchmark-design paper
- methodological negative result
- scoped test of RG-like claims

## Venues Most Compatible With This Version

- physics/stat-mech venues open to careful negative computational results
- ML venues that value benchmark design and rigorous falsification framing
- venues where a well-scoped methodological contribution is preferable to an
  overstated physical breakthrough
