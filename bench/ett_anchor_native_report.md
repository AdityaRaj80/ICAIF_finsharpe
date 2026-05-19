# ETTh1 Native-Head Anchor (PREREG §9a-NATIVE, amendment j)

_Informer split 96->96, native heads + RevIN, d_model=128, seed 0. persistence MSE=1.2944. Published ETTh1 96 ~0.37-0.42 (full tuning); PASS gate <0.55._

| model (native) | test MSE | <pers | <0.55 | <=2x trio-best |
|--|--|--|--|--|
| dlinear | 0.3893 | Y | Y | Y |
| patchtst | 0.3921 | Y | Y | Y |
| itransformer | 0.4183 | Y | Y | Y |

**ANCHOR-NATIVE VERDICT: PASS** (pre-registered amendment j, hash before run). elapsed 57s