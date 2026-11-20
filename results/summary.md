Mean session return (25 tracks) on 200 held-out listeners per profile. Learners: 20000 training episodes, α = 0.02, γ = 0.9, ε 1.0 → 0.05; mean ± 95% CI over 5 training seeds.

| Agent | stable | drifter | abrupt | picky | explorer | **mean** |
|---|---|---|---|---|---|---|
| Q-learning | 12.39 ± 0.53 | 7.34 ± 0.62 | 5.80 ± 0.84 | -0.18 ± 0.53 | 0.47 ± 0.44 | **5.16 ± 0.46** |
| SARSA | 12.51 ± 0.37 | 7.63 ± 0.39 | 5.98 ± 0.20 | 0.09 ± 0.58 | 0.53 ± 0.44 | **5.35 ± 0.15** |
| Q-learning, no belief | 4.85 ± 4.03 | 2.92 ± 1.29 | 0.62 ± 0.91 | -5.19 ± 2.11 | -4.58 ± 0.71 | **-0.28 ± 1.35** |
| random | -11.95 | -10.76 | -12.30 | -14.36 | -13.29 | **-12.53** |
| content similarity (STAY) | -3.87 | -2.65 | -5.50 | -8.57 | -14.37 | **-6.99** |
| skip-streak k=2 | 5.11 | 3.02 | 1.26 | -5.88 | -4.56 | **-0.21** |
| sliding-window bandit | -9.49 | -9.06 | -10.21 | -12.77 | -11.78 | **-10.66** |
| belief greedy (no learning) | 12.55 | 7.61 | 6.41 | -0.23 | 0.38 | **5.34** |
| oracle | 19.77 | 20.73 | 20.89 | 10.08 | 14.41 | **17.18** |
