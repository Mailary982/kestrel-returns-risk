# Kestrel Home — Pre-Dispatch Return Risk

## What this is

A reproducible pre-dispatch return-risk system for Kestrel Home. It scores each recent order by return risk and exposes the score through a small local API and UI so operations can prioritize confirmation calls.

The intended workflow is:

```text
Historical orders
      ↓
Data audit + cleaning
      ↓
Leakage-controlled features
      ↓
XGBoost return-risk model
      ↓
Risk score
      ↓
Intervention threshold
      ↓
Confirmation call
      ↓
Measure actual returns
      ↓
Recalibrate / adjust threshold
```

The score is primarily a **ranking signal**, not a calibrated probability.

## Business decision

The operations policy gives:

| Item | Value |
|---|---:|
| Return processing cost | ₹1,150 / returned order |
| Confirmation call | ₹45 / completed call |
| Historical call prevention assumption | 35% |

An initial pilot operating point is **0.15**.

On the 2,096-order test set, this flags **366 orders (17.46%)**. In the chronological validation set, orders at or above 0.15 had a **28.42% observed return rate**.

Using the stated ₹45 call cost, ₹1,150 return-processing cost and historical 35% prevention assumption, the modeled scenario value at 0.15 is:

- Expected test returns: 104.0
- Expected prevented returns: 36.4
- Confirmation-call cost: ₹16,470
- Gross avoided return-processing cost: ₹41,862
- Modeled net scenario value: **₹25,392**

This is a **scenario estimate, not guaranteed savings**. The threshold should be validated with a controlled intervention pilot.

## Data and outputs

The project expects the supplied files under `data/`:

```text
data/
├── train.csv
├── test_unlabelled.csv
├── customers.csv
├── products.csv
├── sample_submission.csv
└── ops-policy.pdf
```

Client data must remain private and must not be committed to a public repository.

Generated outputs include:

```text
outputs/
├── clean_train.csv
├── clean_test.csv
├── predictions.csv
└── validation_predictions.csv

reports/
├── data_audit.json
├── model_metrics.json
├── feature_importance.csv
├── threshold_analysis.csv
├── intervention_economics.csv
└── intervention_economics_summary.txt

models/
├── model.joblib
├── preprocessor.joblib
└── metadata.json
```

`outputs/predictions.csv` contains one prediction for every test order:

```text
order_id,score
```

Higher scores indicate higher predicted return risk.

## Data quality work

The raw training data contained:

- 11,155 rows
- 10,504 unique order IDs
- 651 duplicate rows
- 1,200 returned orders after deduplication
- 11.42% order-level return rate
- 2,096 test orders

No conflicting return labels were found among duplicate order IDs.

### October 2025 payment anomaly

October 2025 contained a systematic order-value anomaly. Recorded values were approximately 100× the expected product value.

Expected value:

```text
list_price_inr × qty × (1 - discount_pct / 100)
```

The correction was applied only to October 2025 records whose observed/expected ratio was approximately 100.

This corrected 748 raw training rows before deduplication and 700 unique orders after deduplication.

## Leakage control

The prediction is intended to happen before dispatch.

The following fields were excluded:

```text
pickup_scheduled_at
last_service_event_type
```

The operations policy states that after a return is approved, logistics writes `pickup_scheduled_at` and the service system records a `REVERSE_PICKUP` event. Those fields can therefore contain downstream return information.

They were removed rather than allowing the model to learn from information unavailable at the intended decision point.

Raw `customer_id` and raw delivery-note text were also not used directly as model features.

## Features

The model uses structured customer, order, product and delivery information, including:

- prior order and return history
- prior-return indicators and return-rate features
- customer tenure
- Shield membership
- payment mode
- sales channel
- source
- discount
- quantity
- normalized order value
- promised delivery days
- gift status
- product family/model
- list price
- warranty
- product age
- default-pincode indicator
- delivery-note metadata

## Model

A single XGBoost binary classifier is used.

XGBoost was selected because the problem is structured tabular classification with nonlinear interactions among customer, product, payment, channel and order attributes. A single model keeps the implementation reproducible and deployable without unnecessary ensemble complexity.

## Validation

The primary evaluation is an out-of-time chronological split.

```text
Training:   2025-04-01 → 2026-04-01
Validation: 2026-04-02 → 2026-06-30
```

Validation population:

```text
2,101 orders
Validation return rate: 11.52%
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

Accuracy is not treated as the primary business metric because the return class is only about 11%. A high accuracy figure can still hide poor return detection.

## Threshold and intervention economics

The threshold analysis evaluates the trade-off between intervention volume and return concentration.

At **0.15**:

```text
Test orders flagged:       366 / 2,096
Flagged share:             17.46%
Validation return rate:    28.42%
```

The historical 35% prevention figure is used only as a scenario assumption. It has not been demonstrated causally on model-selected customers.

No hold/cancellation cost was supplied, so the economics do not claim savings from cancellation or dispatch holds.

## API and UI

The local service provides:

```text
GET  /health
POST /predict
GET  /demo-order
GET  /
```

`POST /predict` accepts a single order snapshot and returns:

- `order_id`
- `score`
- `risk_band`
- `intervention_recommended`
- human-readable reasons
- a note that the score is primarily a ranking signal

The UI calls the API and displays the prediction in an operations-friendly format.

The service uses the saved local model and does not require a paid API key.

## Run locally

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Install:

```bash
pip install -r requirements.txt
```

Start the service:

```bash
uvicorn app.main:app --reload
```

Open:

```text
http://127.0.0.1:8000
```

API documentation:

```text
http://127.0.0.1:8000/docs
```

## Reproduce the pipeline

From the project root:

```bash
python src/audit.py
python src/clean.py
python src/train.py
python src/test.py
python src/economics.py
```

The training script performs the out-of-time validation. The test script trains the final model on the cleaned historical training set and generates the test predictions.

## Known limitations

1. ROC-AUC and PR-AUC are useful but not strong enough to claim near-perfect prediction.
2. Scores are not calibrated probabilities.
3. Supplied customer-history counts do not always match independently reconstructed temporal histories.
4. Some distribution shift exists between historical and later/test data.
5. The 35% prevention figure is historical and not a causal estimate for this model's selected customers.
6. No explicit hold/cancellation cost was supplied.
7. The system does not estimate heterogeneous treatment effects.
8. The initial 0.15 threshold is a pilot operating point, not a proven optimum.

## Recommended next step

Run a controlled confirmation-call pilot. Log the score, intervention decision, completed call, customer response, actual return, return reason and intervention cost.

Use treatment/control results to estimate the incremental effect of intervention, recalibrate the model if required, and then revisit the threshold and economics.

## Submission note

The client explicitly requires Kestrel data to remain private. Do not publish the supplied CSV/PDF client data in a public repository. Submit through the permitted private-repository or ZIP route.
