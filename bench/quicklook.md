# Phase-2 QUICKLOOK (early directional check, NOT H1 verdict)

_H=5; test-split rebalance points=149; endpoint=hard top-decile equal-weight (cost=10bps); PARTIAL — DSR/PBO/k.6 not computed; ridge may be missing._

| backbone | arm | cells | test_rows | annSharpe | rank-IC |
|--|--|--|--|--|--|
| itransformer | mse | 15 | 125554 | +0.028 | -0.0196 |
| itransformer | risk | 13 | 122977 | +0.115 | +0.0064 |
| patchtst | mse | 0 | 0 |   .   |   .    |
| patchtst | risk | 0 | 0 |   .   |   .    |
| tft | mse | 0 | 0 |   .   |   .    |
| tft | risk | 0 | 0 |   .   |   .    |
| gcformer | mse | 0 | 0 |   .   |   .    |
| gcformer | risk | 0 | 0 |   .   |   .    |
| dlinear | mse | 0 | 0 |   .   |   .    |
| dlinear | risk | 0 | 0 |   .   |   .    |
| lstm | mse | 0 | 0 |   .   |   .    |
| lstm | risk | 0 | 0 |   .   |   .    |
| rnn | mse | 0 | 0 |   .   |   .    |
| rnn | risk | 0 | 0 |   .   |   .    |
| cnn | mse | 0 | 0 |   .   |   .    |
| cnn | risk | 0 | 0 |   .   |   .    |
| ridge | ridge | 0 | 0 |   .   |   .    |

### Paired (risk − mse), partial:
| backbone | dSharpe_vs_mse | dRankIC_vs_mse |
|--|--|--|
| itransformer | +0.087 | +0.0260 |

**WARNING**: this is a DIRECTIONAL CHECK with partial data. It is NOT the H1 verdict. Per PREREG §12 k.6, H1 requires the full eval, Ridge comparator, DSR≥0.95 vs both MSE and Ridge, PBO≤0.5, and is computed only by engine/phase2_aggregate.py on the complete 1275-cell grid.