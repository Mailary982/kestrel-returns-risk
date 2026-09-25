# Kestrel Home — Submission Form

## 1. What did you build, and what business decision does it support? State the number and the rupees.

I built an end-to-end pre-dispatch return-risk system that audits and cleans the historical data, prevents downstream leakage, trains an XGBoost classifier, validates it chronologically, scores every recent test order, and exposes the prediction through a local FastAPI service and UI.

The business decision is which orders should receive a pre-dispatch confirmation call.

The test set contains **2,096 orders**. At the initial pilot threshold of **0.15**, **366 orders (17.46%)** are flagged.

Kestrel's policy gives a **₹1,150 average processing cost per returned order** and a **₹45 confirmation-call cost**. Using the historical **35% prevention** assumption, the 0.15 scenario produces a modeled net value of approximately **₹25,392**.

This is a scenario estimate, not guaranteed savings.

---

## 2. What score do you expect `predictions.csv` to get on the hidden outcomes, on which metric, and why? Say how you estimated it.

I expect the hidden-test **ROC-AUC to be approximately 0.75**, with a reasonable uncertainty range of roughly **0.72–0.77**.

I use ROC-AUC because the submission is a continuous risk score and the assignment defines higher score as higher return likelihood. ROC-AUC evaluates ranking across thresholds rather than depending on one arbitrary classification threshold.

The estimate comes from the out-of-time validation ROC-AUC of **0.7492** on 2,101 later-period orders. The hidden outcomes are unavailable, so this is an estimate rather than a guarantee.

---

## 3. How do you know it works? How did you validate it, on what split, error rate, and the kind of case it gets wrong?

I used an out-of-time chronological split:

```text
Training:   2025-04-01 → 2026-04-01
Validation: 2026-04-02 → 2026-06-30
```

Validation:

```text
2,101 orders
11.52% return rate
```

Results:

| Metric | Result |
|---|---:|
| ROC-AUC | 0.7492 |
| PR-AUC | 0.3663 |
| Accuracy | 89.20% |
| Precision | 63.16% |
| Recall | 14.88% |
| F1 | 24.08% |

The classification error rate at the default 0.5 threshold is **10.80%**, but this is not the main operating metric because the return class is imbalanced.

The model's weaker cases are generally borderline/ranking cases where customer, product, payment and channel signals do not cleanly separate returned from non-returned orders. Recall at 0.5 is only 14.88%, which is why the system is treated as a ranking tool and threshold analysis is performed separately.

---

## 4. Did you change, narrow, or push back on the client's ask? What, when, and why?

Yes.

The client asked for **95%+ accuracy** and proposed holding every flagged order. I did not use 95% accuracy as the primary acceptance criterion.

The historical return rate is only about 11%, so accuracy alone can look high while missing many actual returns. I therefore evaluated ROC-AUC, PR-AUC, precision, recall, F1, threshold-level return concentration and intervention economics.

I also did not make automatic cancellation the default action. The initial recommendation is a confirmation-call pilot because the policy provides an explicit call cost and historical prevention result.

I would not automatically penalize or cancel Shield customers based only on risk score because the policy says Shield members have free returns and the highest lifetime value segment.

---

## 5. What is wrong with what you are handing us, or with the data we handed you?

### Data issues

- 11,155 raw training rows contained 651 duplicate rows.
- October 2025 contained an approximately 100× payment-value anomaly.
- `pickup_scheduled_at` and `last_service_event_type` can contain downstream return information and were not trusted as pre-dispatch predictors.
- Supplied customer-history counts do not always match independently reconstructed temporal histories.
- Some temporal/product/customer/channel distribution shifts exist between historical and later data.

### System limitations

- ROC-AUC 0.7492 is useful but not near-perfect.
- Scores are not calibrated probabilities.
- The historical 35% prevention figure is not a causal estimate for model-selected customers.
- No explicit hold/cancellation cost was supplied.
- The model does not estimate heterogeneous treatment effects.
- The 0.15 threshold is a pilot operating point, not a proven optimum.

---

## 6. What does one prediction cost, and what would a month cost at Kestrel's volume (about 700 orders a month)? Show the arithmetic.

The current implementation uses a local XGBoost model and does not call a paid model API.

Therefore, paid inference/API cost is:

```text
Cost per prediction = ₹0 paid API cost

700 orders/month × ₹0
= ₹0/month in paid inference/API charges
```

This excludes future hosting, monitoring, database/CRM integration, engineering time and the **₹45 operational confirmation-call cost**. The ₹45 call is an intervention cost, not model-inference cost.

---

## 7. What did you deliberately leave out, and why that rather than something else?

I deliberately left out:

### `pickup_scheduled_at`

It can be written after a return is approved, so it is downstream of the decision the model is supposed to predict.

### `last_service_event_type`

Values such as `REVERSE_PICKUP` can reveal downstream return activity.

### Raw `customer_id`

Using it directly would encourage memorization rather than generalizable customer behavior.

### Raw delivery-note text

I used structured metadata from the note rather than raw text so the model remains simple, reproducible and less sensitive to arbitrary wording.

### Hold/cancellation economics

The assignment did not provide a monetary cost for holding/cancelling an order, so I did not invent one.

### Calibrated probabilities / causal treatment effects

Those require additional validation and intervention data and would be misleading to claim from this dataset alone.

---

## 8. Anything you built or found that nobody asked for?

I added:

- a detailed data-quality audit
- explicit payment-anomaly detection
- a leakage audit
- a model-quality audit comparing supplied and reconstructed customer history
- threshold-level intervention economics
- human-readable prediction reasons
- risk bands
- a FastAPI endpoint
- a local UI
- a production-pilot logging plan
- explicit limitations and assumptions

I also documented possible future interventions such as installation/compatibility checks, selected open-box delivery and exchange/store-credit options. These are recommendations for future testing, not implemented interventions.

---

## 9. What did you use AI for? Which tools and models, where did they help, where did they mislead you, what did you throw away.

I used **ChatGPT** as a development assistant for:

- code scaffolding
- feature-engineering discussion
- debugging
- reviewing implementation choices
- documentation
- QA and submission preparation

The actual data cleaning, validation, model training, prediction generation and economics calculations were run locally in Python.

No paid external model API is required by the final Kestrel service.

AI assistance was useful for quickly exploring implementation options, but I discarded or changed suggestions when they conflicted with the actual prediction point, data evidence or assignment requirements. In particular, I did not keep downstream operational fields such as `pickup_scheduled_at` or `last_service_event_type` merely because they improved apparent predictive performance.

---

## 10. Three-minute screen recording

**Recording completed.**

Recording link:

```text
[INSERT RECORDING LINK]
```

---

## 11. GitHub repository

Repository:

```text
[INSERT PRIVATE GITHUB REPOSITORY LINK]
```

The Kestrel client data must not be published in a public repository. The repository should be private or the project should be delivered as a ZIP, consistent with the assignment instructions.

---

## 12. Monday handoff — the three things someone needs to know

### 1. Do not treat 0.15 as a proven optimal threshold.

It is the initial pilot operating point based on out-of-time validation and scenario economics.

### 2. Run a controlled confirmation-call pilot.

Log score, intervention, completed call, customer response and actual return outcome.

### 3. Recalculate the economics after the pilot.

Measure incremental return reduction, actual intervention cost and net savings before expanding the intervention or automating dispatch decisions.

---

## 13. Main deliverables

```text
predictions.csv
README.md
memo.md
submission-form.md
FastAPI service + UI
model artifacts
validation reports
intervention economics
```

The prediction file contains one row per test `order_id` with a continuous return-risk score.
