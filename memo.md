# Kestrel Home — Returns Risk
## One-page memo to Ritu, Head of D2C Operations

### Decision

Use the return-risk model as a **pre-dispatch prioritization tool**, not as an automatic cancellation rule. Start with a controlled confirmation-call pilot for orders scoring **0.15 or higher**.

### The number

The model was evaluated using an out-of-time chronological split:

- Validation orders: **2,101**
- Validation return rate: **11.52%**
- ROC-AUC: **0.7492**
- PR-AUC: **0.3663**

At a 0.15 score threshold, **366 of 2,096 recent test orders (17.46%)** would be flagged. In validation, this group had a **28.42% observed return rate**.

I did not use the requested 95% accuracy as the acceptance criterion. With returns representing only about 11% of orders, accuracy alone can hide missed returns. The model is more useful as a ranking system for deciding where limited intervention capacity should be spent.

### The rupees

Kestrel's policy gives a **₹1,150 average processing cost per returned order** and **₹45 per completed confirmation call**. The historical spring pilot reported approximately **35% prevention of otherwise-occurring returns** among called orders.

Under those assumptions, the 0.15 operating point gives this scenario:

| Item | Scenario value |
|---|---:|
| Test orders flagged | 366 |
| Expected test returns | 104.0 |
| Expected prevented returns | 36.4 |
| Confirmation-call cost | ₹16,470 |
| Gross avoided return-processing cost | ₹41,862 |
| Modeled net scenario value | **₹25,392** |

This is **not guaranteed savings**. The 35% prevention figure is historical and has not been established causally for model-selected customers.

### What I changed

I found **651 duplicate training rows** and removed them at `order_id` level. I also found an October 2025 payment-value anomaly where recorded order values were approximately 100× expected product value; I corrected only the affected October records.

I excluded `pickup_scheduled_at` and `last_service_event_type` because the policy indicates that these can be written after a return has already entered the operational process. Using them would leak downstream information into a pre-dispatch model.

### What I would do next week

1. Run a controlled confirmation-call pilot at the initial 0.15 operating point and record intervention, customer response and actual return outcomes.
2. Measure the **incremental** return reduction caused by the call rather than assuming the historical 35% effect applies unchanged.
3. Recalculate the economics and recalibrate/adjust the model threshold using the pilot results before expanding the intervention.

**Bottom line:** the model provides useful ranking signal and a working operational service, but the next business question is causal: *does intervening on model-selected orders actually reduce returns enough to justify the intervention cost?*
