#!/usr/bin/env python3
"""
Blackjack Probability Simulator
===============================
Monte Carlo analysis of blackjack strategies with NumPy.

Two engines:
  1. Vectorized engine  - plays N hands *in parallel* (infinite-deck model).
                          Used to compare hit/stand strategies and rule variations.
  2. Shoe engine        - sequential, realistic multi-deck shoe with Hi-Lo card
                          counting. Card counting needs memory of dealt cards, so
                          it cannot use the infinite-deck assumption.

Rules modelled: dealer peeks for blackjack, no splitting, no surrender,
double on any first two cards, one card to a doubled hand.

Usage:
    python blackjack_sim.py
    python blackjack_sim.py --trials 50000 --count-hands 200000 --seed 7   # quick run
"""
from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass

import numpy as np

STAND, HIT, DOUBLE = 0, 1, 2
ACTION_NAMES = {STAND: "STAND", HIT: "HIT", DOUBLE: "DOUBLE"}

# Card values: 2-9 face value, 10 (T/J/Q/K) = 10, Ace = 11 (reduced to 1 as needed)
CARD_VALUES = np.arange(2, 12)
CARD_PROBS = np.array([1, 1, 1, 1, 1, 1, 1, 1, 4, 1]) / 13.0  # P(10-value) = 4/13


@dataclass(frozen=True)
class Rules:
    label: str = "3:2, S17"
    bj_payout: float = 1.5
    dealer_hits_soft_17: bool = False


# Basic strategy (hit/stand/double, no splits) built as a lookup table
def _basic_scalar(total: int, soft: bool, up: int) -> tuple[int, int]:
    """Return (preferred action, fallback action if doubling is not allowed)."""
    if soft:
        if total >= 19:
            return STAND, STAND
        if total == 18:
            if up in (3, 4, 5, 6):
                return DOUBLE, STAND
            return (STAND, STAND) if up in (2, 7, 8) else (HIT, HIT)
        if total == 17:
            return (DOUBLE, HIT) if up in (3, 4, 5, 6) else (HIT, HIT)
        if total in (15, 16):
            return (DOUBLE, HIT) if up in (4, 5, 6) else (HIT, HIT)
        if total in (13, 14):
            return (DOUBLE, HIT) if up in (5, 6) else (HIT, HIT)
        return HIT, HIT  # soft 12 (A,A) - no splitting, so just hit
    if total >= 17:
        return STAND, STAND
    if total >= 13:
        return (STAND, STAND) if up <= 6 else (HIT, HIT)
    if total == 12:
        return (STAND, STAND) if 4 <= up <= 6 else (HIT, HIT)
    if total == 11:
        return (DOUBLE, HIT) if up <= 10 else (HIT, HIT)
    if total == 10:
        return (DOUBLE, HIT) if up <= 9 else (HIT, HIT)
    if total == 9:
        return (DOUBLE, HIT) if 3 <= up <= 6 else (HIT, HIT)
    return HIT, HIT


# TABLE[can_double, is_soft, hand_total, dealer_upcard]
TABLE = np.zeros((2, 2, 22, 12), dtype=np.int8)
for _soft in (0, 1):
    for _t in range(4, 22):
        for _up in range(2, 12):
            _pref, _fall = _basic_scalar(_t, bool(_soft), _up)
            TABLE[1, _soft, _t, _up] = _pref
            TABLE[0, _soft, _t, _up] = _fall
TABLE_LIST = TABLE.tolist()  # fast scalar lookups for the shoe engine



# Strategies: f(total, soft, dealer_up, can_double) -> array of actions
def basic_strategy(total, soft, up, can_double):
    return TABLE[int(can_double), (soft > 0).astype(np.int64), np.minimum(total, 21), up]


def threshold_strategy(t: int):
    """Hit until hand total >= t, then stand."""
    def strat(total, soft, up, can_double):
        return np.where(total < t, HIT, STAND)
    return strat


STRATEGIES = {
    "Always stand": threshold_strategy(0),
    "Stand on 12+": threshold_strategy(12),
    "Stand on 15+": threshold_strategy(15),
    "Mimic dealer (17+)": threshold_strategy(17),
    "Stand on 20+": threshold_strategy(20),
    "Basic strategy": basic_strategy,
}



# Engine 1: vectorized, infinite deck

def _draw(rng, n):
    return rng.choice(CARD_VALUES, size=n, p=CARD_PROBS)


def _add_card(total, soft, card, mask=None):
    """Add `card` to hands (where mask is True); demote aces from 11 to 1 as needed."""
    if mask is None:
        mask = np.ones(total.shape, dtype=bool)
    total = total + np.where(mask, card, 0)
    soft = soft + (mask & (card == 11)).astype(np.int64)
    for _ in range(2):  # at most two aces can need demoting after one card
        fix = (total > 21) & (soft > 0)
        total = total - 10 * fix
        soft = soft - fix
    return total, soft


def simulate_hands(n: int, strategy, rules: Rules, rng) -> np.ndarray:
    """Play n independent hands in parallel. Returns profit per hand in units of the initial bet."""
    zeros = np.zeros(n, dtype=np.int64)
    p_total, p_soft = _add_card(zeros, zeros, _draw(rng, n))
    d_up = _draw(rng, n)
    p_total, p_soft = _add_card(p_total, p_soft, _draw(rng, n))
    d_total, d_soft = _add_card(zeros, zeros, d_up)
    d_total, d_soft = _add_card(d_total, d_soft, _draw(rng, n))

    player_bj = p_total == 21
    dealer_bj = d_total == 21
    bet = np.ones(n)
    active = ~(player_bj | dealer_bj)

    # --- player's turn
    for step in range(20):
        if not active.any():
            break
        action = strategy(p_total, p_soft, d_up, step == 0)
        action = np.where(active, action, STAND)
        hit, dbl = action == HIT, action == DOUBLE
        p_total, p_soft = _add_card(p_total, p_soft, _draw(rng, n), hit | dbl)
        bet = np.where(dbl, 2.0, bet)
        active = hit & (p_total <= 21)

    player_bust = p_total > 21
    needs_dealer = ~(player_bj | dealer_bj | player_bust)

    # --- dealer's turn
    for _ in range(20):
        want = needs_dealer & (
            (d_total < 17)
            | (rules.dealer_hits_soft_17 & (d_total == 17) & (d_soft > 0))
        )
        if not want.any():
            break
        d_total, d_soft = _add_card(d_total, d_soft, _draw(rng, n), want)

    # --- settle
    payout = np.zeros(n)
    payout = np.where(player_bj & ~dealer_bj, rules.bj_payout, payout)
    payout = np.where(dealer_bj & ~player_bj, -1.0, payout)
    resolved = ~(player_bj | dealer_bj)
    win = resolved & ~player_bust & ((d_total > 21) | (p_total > d_total))
    lose = resolved & (player_bust | ((d_total <= 21) & (p_total < d_total)))
    payout = np.where(win, bet, payout)
    payout = np.where(lose, -bet, payout)
    return payout



# Engine 2: multi-deck shoe with Hi-Lo counting (sequential)

HILO = np.zeros(12, dtype=np.int64)
HILO[2:7] = 1      # 2-6  -> +1
HILO[10] = -1      # 10s  -> -1
HILO[11] = -1      # aces -> -1


def _hand(cards):
    t, aces = sum(cards), cards.count(11)
    while t > 21 and aces:
        t -= 10
        aces -= 1
    return t, aces > 0


def _play_round(shoe, ptr, rules):
    p = [shoe[ptr], shoe[ptr + 2]]
    d = [shoe[ptr + 1], shoe[ptr + 3]]
    ptr += 4
    pt, ps = _hand(p)
    dt, ds = _hand(d)
    if pt == 21 or dt == 21:
        if pt == 21 and dt == 21:
            return 0.0, ptr
        return (rules.bj_payout if pt == 21 else -1.0), ptr

    up, bet, first = d[0], 1.0, True
    while True:
        a = TABLE_LIST[1 if first else 0][int(ps)][min(pt, 21)][up]
        if a == STAND:
            break
        if a == DOUBLE:
            bet = 2.0
        p.append(shoe[ptr])
        ptr += 1
        pt, ps = _hand(p)
        if pt > 21:
            return -bet, ptr
        if a == DOUBLE:
            break
        first = False

    while dt < 17 or (rules.dealer_hits_soft_17 and dt == 17 and ds):
        d.append(shoe[ptr])
        ptr += 1
        dt, ds = _hand(d)
    if dt > 21 or pt > dt:
        return bet, ptr
    if pt < dt:
        return -bet, ptr
    return 0.0, ptr


def simulate_shoe(n_hands: int, decks: int, penetration: float, rules: Rules, rng):
    """Returns (profit per unit initial bet, true count at deal) for every hand."""
    deck = [v for v in range(2, 10) for _ in range(4)] + [10] * 16 + [11] * 4
    base = np.array(deck * decks)
    size = len(base)
    cut = min(int(size * penetration), size - 20)

    results = np.empty(n_hands)
    true_counts = np.empty(n_hands)
    done = 0
    while done < n_hands:
        shuffled = rng.permutation(base)
        cum = np.concatenate(([0], np.cumsum(HILO[shuffled]))).tolist()
        shoe = shuffled.tolist()
        ptr = 0
        while ptr < cut and done < n_hands:
            decks_left = max((size - ptr) / 52.0, 0.5)
            true_counts[done] = cum[ptr] / decks_left
            results[done], ptr = _play_round(shoe, ptr, rules)
            done += 1
    return results, true_counts


def bet_ramp(true_count, max_bet: int = 8):
    """Bet 1 unit at TC<=1, then 2,4,6,8 units at TC 2,3,4,5+."""
    return np.clip(2 * (np.floor(true_count) - 1), 1, max_bet)



# Statistics + terminal output

def summarize(payout: np.ndarray) -> dict:
    n = len(payout)
    ev = payout.mean()
    se = payout.std(ddof=1) / math.sqrt(n)
    return {
        "n": n, "ev": ev, "se": se,
        "ci": (ev - 1.96 * se, ev + 1.96 * se),
        "win": (payout > 0).mean(), "push": (payout == 0).mean(), "loss": (payout < 0).mean(),
        "std": payout.std(ddof=1),
    }


def bar(frac: float, width: int = 30) -> str:
    return "#" * max(0, int(round(frac * width)))


def header(title: str) -> None:
    print("\n" + "=" * 78)
    print(f" {title}")
    print("=" * 78)


def report_strategies(args, rules: Rules, seeds) -> dict:
    header(f"1. STRATEGY COMPARISON  ({args.trials:,} hands each, rules: {rules.label}, infinite deck)")
    print(f"{'Strategy':<20}{'Win%':>7}{'Push%':>7}{'Loss%':>7}{'EV/hand':>10}{'House edge':>12}   95% CI (EV)")
    print("-" * 78)
    out, payouts = {}, {}
    for (name, strat), seed in zip(STRATEGIES.items(), seeds):
        pay = simulate_hands(args.trials, strat, rules, np.random.default_rng(seed))
        s = summarize(pay)
        out[name], payouts[name] = s, pay
        print(f"{name:<20}{s['win']*100:>7.2f}{s['push']*100:>7.2f}{s['loss']*100:>7.2f}"
              f"{s['ev']:>+10.4f}{-s['ev']*100:>11.2f}%   [{s['ci'][0]:+.4f}, {s['ci'][1]:+.4f}]")

    print("\nHouse edge by strategy (longer bar = worse for the player):")
    worst = max(-s["ev"] for s in out.values())
    for name, s in out.items():
        print(f"  {name:<20} {bar(-s['ev'] / worst):<30} {-s['ev']*100:5.2f}%")

    best = "Basic strategy"
    print(f"\nPayout distribution, {best} (units of initial bet):")
    pay = payouts[best]
    for v in (-2.0, -1.0, 0.0, 1.0, 1.5, 2.0):
        frac = (pay == v).mean()
        print(f"  {v:+5.1f} | {bar(frac / 0.5, 40):<40} {frac*100:6.2f}%")

    print(f"\nEV convergence, {best} (running mean EV/hand as trials grow):")
    cum = np.cumsum(pay) / np.arange(1, len(pay) + 1)
    marks = [m for m in (100, 1_000, 10_000, 50_000, 100_000, 500_000, 1_000_000) if m < len(pay)] + [len(pay)]
    for m in marks:
        print(f"  n={m:>9,}  EV = {cum[m-1]:+.4f}")
    return out


def report_rules(args, seeds) -> None:
    header(f"2. VARYING GAME CONDITIONS  (Basic strategy, {args.trials:,} hands each)")
    variants = [
        Rules("3:2 payout, dealer stands S17", 1.5, False),
        Rules("3:2 payout, dealer hits  H17", 1.5, True),
        Rules("6:5 payout, dealer stands S17", 1.2, False),
        Rules("6:5 payout, dealer hits  H17", 1.2, True),
    ]
    print(f"{'Rule set':<34}{'EV/hand':>10}{'House edge':>12}   95% CI")
    print("-" * 78)
    for r, seed in zip(variants, seeds):
        s = summarize(simulate_hands(args.trials, basic_strategy, r, np.random.default_rng(seed)))
        print(f"{r.label:<34}{s['ev']:>+10.4f}{-s['ev']*100:>11.2f}%   [{s['ci'][0]:+.4f}, {s['ci'][1]:+.4f}]")


def report_counting(args, rules: Rules, seed) -> None:
    header(f"3. HI-LO CARD COUNTING  ({args.count_hands:,} hands, {args.decks} decks, "
           f"{args.penetration:.0%} penetration, 1-8 unit bet spread)")
    rng = np.random.default_rng(seed)
    t0 = time.time()
    r, tc = simulate_shoe(args.count_hands, args.decks, args.penetration, rules, rng)
    print(f"(shoe simulation took {time.time() - t0:.1f}s)\n")

    # Edge by true count: the core evidence that counting works
    print("Player edge by true count (flat 1-unit bets, basic strategy):")
    print(f"  {'True count':<12}{'% of hands':>11}{'EV/hand':>10}{'Edge':>9}")
    bucket = np.clip(np.floor(tc), -2, 5).astype(int)
    for b in range(-2, 6):
        m = bucket == b
        if m.sum() < 50:
            continue
        label = f"<= {b}" if b == -2 else (f">= {b}" if b == 5 else f"{b:+d}")
        ev = r[m].mean()
        print(f"  {label:<12}{m.mean()*100:>10.1f}%{ev:>+10.4f}{ev*100:>+8.2f}%  {'+' * int(max(0, ev) * 200)}")

    bets = bet_ramp(tc)
    flat, counter = summarize(r), summarize(bets * r)
    roi = (bets * r).sum() / bets.sum()
    print("\nFlat betting vs. counting with bet ramp:")
    print(f"  {'':<22}{'Profit/hand':>13}{'Avg bet':>9}{'Edge (per unit wagered)':>26}")
    print(f"  {'Flat bet, basic strat':<22}{flat['ev']:>+13.4f}{1.0:>9.2f}{flat['ev']*100:>+25.2f}%")
    print(f"  {'Hi-Lo + bet ramp':<22}{counter['ev']:>+13.4f}{bets.mean():>9.2f}{roi*100:>+25.2f}%")
    print(f"  Counter profit/hand 95% CI: [{counter['ci'][0]:+.4f}, {counter['ci'][1]:+.4f}]  "
          f"(std/hand = {counter['std']:.2f} units)")
    print("  Note: counting wins more per hand but with far higher variance.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Blackjack Monte Carlo simulator")
    ap.add_argument("--trials", type=int, default=1_000_000, help="hands per strategy (vectorized engine)")
    ap.add_argument("--count-hands", type=int, default=3_000_000, help="hands for the card-counting simulation")
    ap.add_argument("--decks", type=int, default=6)
    ap.add_argument("--penetration", type=float, default=0.75)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Independent, reproducible RNG stream per experiment
    seeds = np.random.SeedSequence(args.seed).spawn(len(STRATEGIES) + 4 + 1)
    base_rules = Rules()
    print(f"Blackjack Probability Simulator | seed={args.seed} | NumPy {np.__version__}")
    report_strategies(args, base_rules, seeds[: len(STRATEGIES)])
    report_rules(args, seeds[len(STRATEGIES): len(STRATEGIES) + 4])
    report_counting(args, base_rules, seeds[-1])
    print()


if __name__ == "__main__":
    main()
