# Erratum — one-sided claims were registered and scored two-sided

_Found 2026-07-25, after the 5-seed results were published. Recorded here rather
than by editing `claims.json`: a pre-registration that gets quietly rewritten
after results exist is worthless, so the original registration stands unchanged
and this document states what was wrong with it._

## What was wrong

Two of the seven claims are **one-sided** in the paper's own wording, and both
were registered as two-sided point-plus-tolerance:

| claim | the paper's words | registered as | should have been |
|---|---|---|---|
| C2 | "improving by **more than** 0.3 percentage points" | `0.3 ± 0.3` (two-sided) | lower bound at 0.3 |
| C6 | "a test accuracy improvement of **up to** 0.35 percentage points" | `0.35 ± 0.3` (two-sided) | upper bound at 0.35 |

Two-sided scoring of a one-sided claim is wrong in both directions. For C2 it
invents an **upper** bound the paper never stated: an observed +0.9 pp plainly
satisfies "more than 0.3", but the registered window `[0.0, 0.6]` scores it
NOT_REPRODUCED. For C6 it invents a **lower** bound: the paper says the
improvement reaches *at most* 0.35, so a small positive value is consistent with
it, but the registered window `[0.05, 0.65]` would reject +0.02.

C3 ("returning to the level of the original network") is genuinely two-sided and
was registered correctly.

## What it changed in the published results: nothing

Rescored with the same tolerances under the correct direction:

| claim | 5-seed mean | as published (two-sided) | corrected | changed? |
|---|---|---|---|---|
| C2 | 0.212 | REPRODUCED, window `[0.0, 0.6]` | REPRODUCED, window `[0.0, ∞)` | no |
| C6 | 0.278 | REPRODUCED, window `[0.05, 0.65]` | REPRODUCED, window `(−∞, 0.65]` | no |

Both observations sit inside both windows, so the headline result — **6 of 7
claims replicated on the 5-seed mean** — is unaffected, and C1 remains the only
claim outside its tolerance.

This is stated plainly because the temptation runs the other way: a defect that
changes nothing is easy to leave unreported, and a bench whose whole premise is
publishing inconvenient results does not get to do that.

## Why it still matters

The registration was correct by luck, not by construction. Under a different
draw — a seed mean of 0.9 pp on C2, entirely plausible given the per-seed spread
of −0.05 to 0.39 — the tool would have reported a **false failure** against a
claim the paper's own text was satisfied by. Pre-registration protects against
moving the goalposts after seeing data; it does not protect against putting the
goalposts in the wrong shape beforehand.

## What was fixed

`paper-repro-gym` now supports `tolerance_kind: "lower_bound" | "upper_bound" |
"interval"` alongside the two-sided `"abs"`/`"rel"`, with a regression test that
pins this exact case: an observation of 0.9 against "more than 0.3" must pass,
and the old two-sided scoring must be shown to reject it.

Claims registered from this point on carry the direction the paper actually
used. The CIFAR half's registration is being reviewed for the same defect before
its results are scored.

## Credit

Found by an adversarial review pass over the claim set, not by the author.
