# ETTh1 All-8 Native Anchor (PREREG §12 k.5)

_Native heads + RevIN, d_model=128, seed 0; gates = 1.5x cited-worst-published (mechanically derived, fixed a-priori). persistence=1.2944._

| model | MSE | gate | <gate | <pers | <=2x best |
|--|--|--|--|--|--|
| dlinear | 0.3893 | 0.6 | Y | Y | Y |
| patchtst | 0.3921 | 0.62 | Y | Y | Y |
| itransformer | 0.4183 | 0.59 | Y | Y | Y |
| gcformer | 0.5743 | 0.68 | Y | Y | Y |
| tft | 0.4338 | 0.9 | Y | Y | Y |
| cnn | 0.4977 | 0.83 | Y | Y | Y |
| lstm | 0.4196 | 1.05 | Y | Y | Y |
| rnn | 0.4001 | 1.05 | Y | Y | Y |

**ALL-8 ANCHOR VERDICT: PASS** (gates pre-registered PREREG §12 k.5, hash amendment k, BEFORE this run). elapsed 155s