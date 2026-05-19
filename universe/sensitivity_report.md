# Universe Threshold Sensitivity

_Cutoffs pre-registered as ABSOLUTE economic/data-hygiene floors, NOT outcome-tuned quantiles. Percentiles below are descriptive._

## Baseline (exact build constants) -> **Tier-1 = 870** (reproduces build: OK)

| floor | value | rationale | sits at pctile of eligible pool |
|--|--|--|--|
| median ADV | >=$5,000,000 | standard tradable-liquidity floor | ~p47 |
| total news | >=500 | min for non-degenerate decay series | ~p42 |
| hist start | <=2011-01-01 | 2y warmup before 2013 train | fixed |
| price end | >=2023-12-20 | covers test end | fixed |
| n_rows | >=2800 | ~0.85x expected trading days | fixed |
| max gap | <=10d | no >1wk data holes | fixed |

## 1-D perturbation (|Tier1|, Jaccard vs 870-baseline)

| change | |Tier1| | Jaccard |
|--|--|--|
| $vol>=$2,000,000 | 980 | 0.888 |
| $vol>=$3,000,000 | 918 | 0.948 |
| $vol>=$5,000,000 | 870 | 1.0 |
| $vol>=$7,000,000 | 806 | 0.926 |
| $vol>=$10,000,000 | 725 | 0.833 |
| $vol>=$15,000,000 | 628 | 0.722 |
| news>=300 | 911 | 0.955 |
| news>=400 | 888 | 0.98 |
| news>=500 | 870 | 1.0 |
| news>=750 | 818 | 0.94 |
| news>=1000 | 746 | 0.857 |
| news>=1500 | 609 | 0.7 |
| n_rows>=2400 | 870 | 1.0 |
| n_rows>=2600 | 870 | 1.0 |
| n_rows>=2800 | 870 | 1.0 |
| n_rows>=3000 | 870 | 1.0 |
| n_rows>=3200 | 870 | 1.0 |
| gap<=7 | 870 | 1.0 |
| gap<=8 | 870 | 1.0 |
| gap<=10 | 870 | 1.0 |
| gap<=12 | 870 | 1.0 |
| gap<=15 | 870 | 1.0 |
| hist<=2010-01-01 | 844 | 0.97 |
| hist<=2011-01-01 | 870 | 1.0 |
| hist<=2012-01-01 | 888 | 0.98 |

## Joint perturbation (liquidity x news)

| perturbation | |Tier1| | Jaccard |
|--|--|--|
| $vol>=3,000,000,news>=400 | 940 | 0.926 |
| $vol>=3,000,000,news>=500 | 918 | 0.948 |
| $vol>=3,000,000,news>=750 | 859 | 0.898 |
| $vol>=5,000,000,news>=400 | 888 | 0.98 |
| $vol>=5,000,000,news>=500 | 870 | 1.0 |
| $vol>=5,000,000,news>=750 | 818 | 0.94 |
| $vol>=7,000,000,news>=400 | 818 | 0.914 |
| $vol>=7,000,000,news>=500 | 806 | 0.926 |
| $vol>=7,000,000,news>=750 | 766 | 0.88 |

## Verdict
- Reproduces the build (Tier-1=870).
- One-economic-step-each-side joint Jaccard >= **0.880**; full grid min Jaccard 0.880, size band [766,940].
- Floors are economically pre-registered; Phase-2 headline will be reported on the +/-1-step universe band as a robustness row, so the result cannot hinge on a particular cutoff.