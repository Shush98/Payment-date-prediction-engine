# B2B Payment Optimization: From Prediction to Decision Intelligence

## Executive Summary

This project transforms a standard "predict payment date" ML problem into a **decision intelligence system** that answers the questions businesses actually care about:

1. **Which customers should we contact today?** (prioritization under capacity constraints)
2. **What's the expected ROI of each collection action?** (cost-benefit optimization)
3. **Does contacting customers actually accelerate payment?** (causal inference)

The shift from *prediction accuracy* to *business impact* is the core differentiator.

---

## Business Context

### The Problem Space

B2B companies face a persistent cash flow challenge: customers pay late.

**Scale of the problem:**
- Average Days Sales Outstanding (DSO) in B2B: 40-50 days
- Late payments affect 87% of businesses (Atradius Payment Practices Barometer)
- Working capital cost: 5-15% annually depending on company credit profile
- In this dataset: **15% of invoices are paid 4+ days late**, with outliers exceeding 200 days

**The collection team's reality:**
- Limited capacity: N calls/emails per day
- Relationship cost: Aggressive collection damages customer relationships
- Information asymmetry: Not all late-payers respond to intervention equally

### The Naive Approach (and Why It Fails)

**What most data scientists build:**
```
Model: Predict days until payment
Evaluation: RMSE = 4.2 days, ±3 day accuracy = 78%
Recommendation: "Here are the predictions, figure out what to do with them"
```

**Why this fails the business:**

| Problem | Impact |
|---------|--------|
| No prioritization | Collection team calls customers randomly |
| No cost-benefit | A $500 invoice gets same attention as $500K |
| Symmetric loss | Model treats "5 days early" same as "5 days late" |
| No uncertainty | Point predictions hide risk |
| No causal thinking | Assumes intervention works equally for everyone |

---

## Reframing: The Real Questions

### Question 1: Who should we contact?

This is a **constrained optimization problem**, not a prediction problem.

**Given:**
- Prediction: Customer X will pay in 25 days (due in 15 days = 10 days late)
- Invoice value: $50,000
- Collection capacity: 20 calls/day
- Call success rate: Varies by customer segment

**The question isn't** "will they be late?" (we know they will be)

**The question is** "is Customer X a better use of our limited collection time than Customer Y?"

### Question 2: What's the expected value of each action?

```
E[Value of Call] = P(call accelerates payment) × Days_Accelerated × Daily_Cost_of_Capital × Invoice_Value
                   - Cost_of_Call 
                   - Relationship_Damage_Risk
```

This requires:
- Probability estimates (not point predictions)
- Causal estimates of intervention effect
- Business cost parameters

### Question 3: Does calling actually work?

**The causal challenge:**

Observed: Customers who receive collection calls pay 5 days faster than those who don't.

**But this is confounded:**
- Responsive customers answer calls AND pay on time
- Large invoices get more attention AND have more payment urgency
- "Good" customers get flagged early AND have better relationships

**We need to isolate the causal effect of intervention** using:
- Natural experiments (capacity constraints, timing quirks)
- Difference-in-differences (policy changes over time)
- Regression discontinuity (threshold-based calling rules)

---

## Data Overview

**Source:** B2B invoice data from a mid-market company

| Metric | Value |
|--------|-------|
| Total invoices | 50,000 |
| Closed (paid) | 40,000 |
| Open (pending) | 10,000 |
| Unique customers | 1,099 |
| Invoice range | $0.72 - $668,593 |
| Time period | 2018-2020 |

**Payment behavior:**
- 37% paid early (before due date)
- 48% paid on time (0-3 days)
- 13% paid late (4-30 days)
- 2% paid very late (30+ days)

**Key insight:** Customer behavior is highly consistent. Top customer averages -2.4 days (early) with low variance. Problem customers (e.g., CCU013) average +42 days late consistently.

---

## Analytical Approach

### Phase 1: Enhanced Prediction Model

- Predict **days late** relative to due date (not raw payment date)
- Output **prediction intervals**, not point estimates
- Incorporate customer payment history as features
- Evaluate on business-relevant metrics (not just RMSE)

### Phase 2: Decision Framework

Build a recommendation engine that:
1. Calculates expected value of intervention for each invoice
2. Ranks invoices by intervention ROI
3. Accounts for collection capacity constraints
4. Outputs actionable daily work queues

### Phase 3: Causal Analysis

Investigate:
1. **Seasonality as instrument:** December shows 2-day improvement (year-end pressure) — can we use seasonal variation to identify responsive vs. unresponsive customers?
2. **Policy discontinuity:** Feb 2020 shows sudden improvement — was there a policy change we can exploit?
3. **Threshold effects:** Do customers behave differently around payment term thresholds?

---

## Success Metrics

**Model metrics (secondary):**
- RMSE, MAE
- Calibration of prediction intervals

**Business metrics (primary):**
- Reduction in average days late
- $ value of accelerated cash flow
- Collection team efficiency (outcomes per call)
- Customer relationship preservation

**Causal metrics:**
- Estimated treatment effect of collection intervention
- Confidence interval of causal estimates
- Identification of customer segments where intervention is effective

---

## Deliverables

1. **Prediction Model:** XGBoost with uncertainty quantification
2. **Decision Engine:** Priority ranking system with ROI calculations
3. **Causal Analysis:** Natural experiment exploitation for treatment effect estimation
4. **Simulation:** What-if analysis for different collection strategies
5. **Documentation:** This problem framing + technical notebooks

---

## Key Insight

> **The goal is not to predict when customers will pay.
> The goal is to decide which customers are worth the effort to contact.**

This reframing transforms a commodity ML project into a strategic decision-support system.
