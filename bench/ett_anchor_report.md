# ETTh1 External-Validity Anchor (PREREG §9a)

_Informer split, 96->96, 7 vars, d_model=128, seed 0; impostor/bug detector, NOT SOTA. persistence MSE=1.2944._

| model | test MSE | test MAE | epochs | <pers | <0.80 | <=2.5x DLinear |
|--|--|--|--|--|--|--|
| dlinear | 0.8897 | 0.7133 | 8 | Y | N | Y |
| patchtst | 1.0005 | 0.7720 | 5 | Y | N | Y |
| itransformer | 1.1089 | 0.8416 | 5 | Y | N | Y |
| gcformer | 0.8475 | 0.6739 | 12 | Y | N | Y |
| tft | 0.9275 | 0.7559 | 7 | Y | N | Y |
| cnn | 1.0960 | 0.8113 | 4 | Y | N | Y |
| lstm | 0.8998 | 0.7493 | 6 | Y | N | Y |
| rnn | 0.7020 | 0.6384 | 6 | Y | Y | Y |

**ANCHOR VERDICT: FAIL (['dlinear', 'patchtst', 'itransformer', 'gcformer', 'tft', 'cnn', 'lstm'])** (criteria pre-registered PREREG §9a, hash amendment i). LSTM/RNN exempt from the 2.5x rule (intrinsically weak baselines, disclosed). elapsed 198s