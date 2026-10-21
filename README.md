# Blackjack Probability Simulator

A Monte Carlo simulation of blackjack in Python and NumPy. It estimates win probabilities, expected value (EV) and house edge across several player strategies and rule variations, and models Hi-Lo card counting on a multi-deck shoe.

## Features

- **Vectorized engine:** plays N hands in parallel with NumPy (infinite-deck model). Used to compare strategies and rule sets.
- **Shoe engine:** sequential multi-deck shoe with Hi-Lo counting, true count and a 1-8 unit bet ramp.
- **Statistics:** win/push/loss rates, EV per hand, house edge, and 95% confidence intervals.
- **Terminal output:** strategy comparison table, house edge bar chart, payout distribution, EV convergence, and edge by true count.
- **Reproducible:** every experiment uses its own seeded random stream (`--seed`).

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python blackjack_sim.py
```

Quick run: `python blackjack_sim.py --trials 50000 --count-hands 200000`

| Option | Default | Meaning |
|---|---|---|
| `--trials` | 1,000,000 | Hands per strategy (vectorized engine) |
| `--count-hands` | 3,000,000 | Hands in the card-counting simulation |
| `--decks` | 6 | Decks in the shoe |
| `--penetration` | 0.75 | Fraction of the shoe dealt before reshuffling |
| `--seed` | 42 | Random seed |

## Rules modelled

Dealer peeks for blackjack, double on any first two cards (one card only), blackjack pays 3:2 unless stated, dealer stands on soft 17 unless stated. **Not modelled:** splitting, surrender, insurance.

## Results (1M hands per strategy, seed 42)

| Strategy | House edge |
|---|---|
| Always stand | 16.04% |
| Stand on 12+ | 8.03% |
| Stand on 15+ | 5.15% |
| Mimic dealer (hit below 17) | 5.69% |
| Hit until 20 | 33.30% |
| **Basic strategy** | **1.00%** |

Rule variations (basic strategy):

| Rules | House edge |
|---|---|
| 3:2 payout, dealer stands on soft 17 | 1.03% |
| 3:2 payout, dealer hits soft 17 | 1.12% |
| 6:5 payout, dealer stands on soft 17 | 2.34% |
| 6:5 payout, dealer hits soft 17 | 2.62% |

Hi-Lo counting (6 decks, 75% penetration, 3M hands, 1-8 unit bet ramp):

| | Edge per unit wagered |
|---|---|
| Flat bet, basic strategy | -0.93% |
| Hi-Lo with bet ramp | -0.29% |

Player edge rises steadily with the true count. Counting cuts the house edge by about 0.6 percentage points here, but it stays negative because this version has no splitting and no count-based strategy deviations.

## How it works

- Hand totals track aces as 11 and demote them to 1 as needed (soft/hard hands).
- Basic strategy is a lookup table indexed by hand total, soft/hard, dealer upcard and whether doubling is allowed.
- Edge results are reported in units of the initial bet, with 95% confidence intervals from the standard error.

## Possible extensions

Splitting and double-after-split, Hi-Lo strategy deviations, Kelly bet sizing, risk of ruin and bankroll simulations, Numba acceleration, unit tests.
