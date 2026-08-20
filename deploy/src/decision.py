import numpy as np

# --- Business parameters (adjustable) ---
DAILY_CAPITAL_RATE = 0.0003   # cost of capital per day (~10 % annually)
CALL_COST          = 15.0     # cost of one collection call in dollars
DAYS_ACCELERATED   = 3.0      # days a successful call typically speeds up payment
DAILY_CAPACITY     = 20       # max calls the collections team can make per day

# --- Action tier thresholds ---
CALL_DELAY_DAYS   = 7        # delay > 7 days → always CALL regardless of amount
CALL_AMOUNT_USD   = 50_000   # delay ≥ 3 days + amount > $50K → escalate to CALL
REMIND_DELAY_DAYS = 3        # delay ≥ 3 days (lower amount) → email REMINDER
# delay 1–2 days → WATCH; delay ≤ 0 → OK


def _p_late(lower, upper):
    """Probability that an invoice will be paid late, derived from the prediction interval."""
    if upper < 0:
        return 0.05
    if lower > 5:
        return 0.95
    width = max(upper - lower, 1.0)
    return float(np.clip(upper / width, 0.0, 1.0))


def _p_responds(cust_std_days_late):
    """Customer responsiveness — inverse of payment variability, clipped to [0.2, 0.8]."""
    return float(np.clip(1.0 - cust_std_days_late / 20.0, 0.2, 0.8))


def expected_value(amount, lower, upper, cust_std):
    """Expected dollar value of making one collection call on this invoice.

    Formula: P(late) * P(responds) * days_accelerated * capital_rate * amount - call_cost
    """
    p_l     = _p_late(lower, upper)
    p_r     = _p_responds(cust_std)
    benefit = p_l * p_r * DAYS_ACCELERATED * DAILY_CAPITAL_RATE * abs(amount)
    return round(benefit - CALL_COST, 2)


def _assign_tier(days_late_ceil, amount):
    """Assign action tier from ceiling-rounded predicted delay and invoice amount.

    Tiers:
        CALL   — delay > 7d (any amount) OR delay 3–7d + amount > $50K
        REMIND — delay 3–7d + amount ≤ $50K
        WATCH  — delay 1–2d (low risk, flag for future monitoring)
        OK     — predicted on time (delay ≤ 0)
    """
    amt = abs(amount)
    if days_late_ceil <= 0:
        return "OK"
    if days_late_ceil > CALL_DELAY_DAYS:
        return "CALL"
    if days_late_ceil >= REMIND_DELAY_DAYS and amt > CALL_AMOUNT_USD:
        return "CALL"
    if days_late_ceil >= REMIND_DELAY_DAYS:
        return "REMIND"
    return "WATCH"  # 1–2 days


def build_priority_table(df):
    """Assign action tiers to open invoices and sort by urgency.

    Tier assignment uses ceiling-rounded delay + invoice amount.
    Within the CALL tier, only the top DAILY_CAPACITY invoices by expected value
    are kept as CALL; overflow invoices are downgraded to REMIND.

    Sort order: CALL → REMIND → WATCH → OK, with EV descending within each tier.

    Returns df with new columns: expected_value, action, rank.
    """
    out = df.copy()

    out["expected_value"] = [
        expected_value(
            row["total_open_amount"],
            row["days_late_lower"],
            row["days_late_upper"],
            row["cust_std_days_late"],
        )
        for _, row in out.iterrows()
    ]

    days_ceil = np.ceil(out["days_late_pred"]).astype(int)
    out["action"] = [
        _assign_tier(d, a)
        for d, a in zip(days_ceil, out["total_open_amount"])
    ]

    # Cap CALL tier at DAILY_CAPACITY — extras become REMIND
    call_idx    = out[out["action"] == "CALL"].sort_values("expected_value", ascending=False).index
    demote_idx  = call_idx[DAILY_CAPACITY:]
    out.loc[demote_idx, "action"] = "REMIND"

    # Sort: CALL → REMIND → WATCH → OK, EV descending within each tier
    tier_order = {"CALL": 0, "REMIND": 1, "WATCH": 2, "OK": 3}
    out["_tier"] = out["action"].map(tier_order)
    out = (out.sort_values(["_tier", "expected_value"], ascending=[True, False])
              .drop(columns=["_tier"])
              .reset_index(drop=True))
    out["rank"] = out.index + 1

    return out
