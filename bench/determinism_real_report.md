# Real-Path Determinism (re-jury-6 MAJOR)

_2 epochs, real loader+AMP+GradScaler+Adam+composite loss, dev=cuda, seed 0 x2 -> bit-identical final weights._

| backbone | params compared | max_absdiff | bit-identical |
|--|--|--|--|
| itransformer | 54 | 0.00e+00 | YES |
| cnn | 24 | 0.00e+00 | YES |

**REAL-DETERMINISM: PASS** (real Phase-2 path; elapsed 29s)