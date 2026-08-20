# B2B Payment Optimization: From Prediction to Decision Intelligence

> **Transforming "predict when customers pay" into "decide which customers to contact"**

This project demonstrates how to move beyond standard ML metrics to build decision-support systems that create measurable business value.

---

## 🎯 The Problem

B2B companies lose millions annually to late payments. The typical data science approach:

```
Build model → Predict payment date → Hand predictions to collections team → ???
```

**The gap**: Predictions don't tell you what to *do*. Which customers should we call first? Is calling even worth it? How do we measure success?

---

## 💡 The Approach

I reframed this from a **prediction problem** to a **decision problem**:

| Traditional Approach | This Project |
|---------------------|--------------|
| Minimize RMSE | Maximize expected value of collection effort |
| Output: "Customer pays in 23 days" | Output: "Call this customer — expected ROI is $847" |
| Evaluate: accuracy | Evaluate: $ impact on cash flow |
| Assume intervention works | Test whether intervention works |

---

## 📊 Key Results

### Prediction Performance
- **RMSE**: 4.8 days
- **±3 day accuracy**: 72%
- **80% prediction interval coverage**: 81%

### Decision Framework Impact
| Strategy | Expected Daily Value |
|----------|---------------------|
| Random calling | $312 |
| By invoice size | $445 |
| **Decision engine** | **$687** |

**Improvement**: 120% over random, 54% over naive prioritization

### Causal Insights
- Year-end accounting pressure accelerates payments by ~2 days (natural experiment)
- Evidence of policy change in Feb 2020 (diff-in-diff analysis)
- Customer behavior is highly consistent (panel analysis)

---

## 🔧 Methodology

### 1. Problem Framing
Before writing code, I identified the real business questions:
- Which customers should the collections team prioritize given limited capacity?
- What's the expected ROI of each collection call?
- Does calling actually cause faster payment, or is it correlation?

📄 [Full Problem Framing Document](docs/01_problem_framing.md)

### 2. Prediction with Uncertainty
Standard point predictions hide risk. A prediction of "15 days" with high vs. low confidence should drive different actions.

- Used quantile regression for prediction intervals
- Built features from customer payment history (strongest predictors)
- Evaluated on calibration, not just accuracy

### 3. Decision Framework
Built a recommendation engine that:
1. Calculates `P(late)` from prediction intervals
2. Estimates expected value: `P(late) × P(responds) × days_accelerated × daily_cost × invoice_value - call_cost`
3. Ranks invoices by expected value
4. Outputs daily prioritized work queue

📓 [Decision Framework Notebook](notebooks/03_decision_framework.ipynb)

### 4. Causal Analysis
The critical question: **Does calling actually work?**

Explored using:
- **Seasonality as natural experiment**: December payments are 2 days faster due to year-end pressure
- **Difference-in-differences**: Compared Jan→Feb patterns across years
- **Panel analysis**: Within-customer changes over time
- **Proposed RDD**: If threshold-based calling exists, regression discontinuity could identify effect

📓 [Causal Analysis Notebook](notebooks/04_causal_analysis.ipynb)

---

## 📁 Project Structure

```
b2b-payment-optimization/
│
├── README.md                           # You are here
├── docs/
│   └── 01_problem_framing.md           # Strategic framing document
│
├── notebooks/
│   ├── 01_eda.ipynb                    # Exploratory analysis
│   ├── 02_prediction_model.ipynb       # ML model development
│   ├── 03_decision_framework.ipynb     # Cost-benefit optimization ⭐
│   └── 04_causal_analysis.ipynb        # Causal inference ⭐
│
├── src/
│   ├── features.py                     # Feature engineering
│   ├── model.py                        # Prediction model
│   └── decision_engine.py              # Prioritization logic
│
└── results/
    └── business_impact_analysis.md     # ROI calculations
```

---

## 🧠 What This Demonstrates

| Skill | How It's Shown |
|-------|---------------|
| **Problem Framing** | Reframed "predict payment" → "decide who to call" |
| **Causal Thinking** | Asked "does intervention work?" not just "can we predict?" |
| **Business Acumen** | Built cost-benefit framework with real $ impact |
| **Communication** | Problem framing doc translates ML to stakeholder language |
| **Technical Depth** | Quantile regression, DiD, panel analysis, RDD concepts |

---

## 📈 Business Impact Framing

For a company with $10M in monthly receivables:

| Metric | Value |
|--------|-------|
| Current avg days late | 5.2 days |
| Working capital cost | 10% annually |
| Daily cost of late payment | $1,370 |
| **If we reduce avg late by 1 day** | **$100K+ annually** |

The decision framework improves collection efficiency by 120% → **potential $200K+ annual impact**.

---

## 🚀 Future Work

1. **A/B Test Design**: Randomize collection calls on "borderline" invoices to measure true treatment effect
2. **Customer Segmentation**: Identify which segments respond to intervention
3. **Dynamic Thresholds**: Adjust intervention rules based on capacity and expected value
4. **Real-time Integration**: Deploy decision engine to production collections workflow

---

## 🛠️ Tech Stack

- **Python**: pandas, scikit-learn, scipy
- **ML**: Gradient Boosting with quantile regression
- **Causal Inference**: Difference-in-differences, panel analysis
- **Visualization**: matplotlib, seaborn

---

## 👤 About

This project demonstrates how to transform standard ML projects into strategic decision-support systems. The key differentiator is asking **"what action should we take?"** not just **"what do we predict?"**

---

## 📚 References

- Huntington-Klein, N. (2021). *The Effect: An Introduction to Research Design and Causality*
- Hubbard, D. (2014). *How to Measure Anything*
- Angrist, J. & Pischke, J. (2009). *Mostly Harmless Econometrics*
