from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "models" / "xgboost_return_risk_final.joblib"
PREPROCESSOR_PATH = ROOT / "models" / "preprocessor_final.joblib"

CUSTOMERS_PATH = ROOT / "data" / "customers.csv"
PRODUCTS_PATH = ROOT / "data" / "products.csv"
TRAIN_PATH = ROOT / "data" / "train.csv"
TEST_PATH = ROOT / "data" / "test_unlabelled.csv"


# ============================================================
# LOAD MODEL + DATA
# ============================================================

model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)

customers = pd.read_csv(CUSTOMERS_PATH)
products = pd.read_csv(PRODUCTS_PATH)
train_raw = pd.read_csv(TRAIN_PATH)
test_raw = pd.read_csv(TEST_PATH)


# ============================================================
# DATETIME NORMALIZATION
# ============================================================

train_raw["order_placed_at"] = pd.to_datetime(
    train_raw["order_placed_at"],
    errors="coerce"
)

test_raw["order_placed_at"] = pd.to_datetime(
    test_raw["order_placed_at"],
    errors="coerce"
)

train_raw["pickup_scheduled_at"] = pd.to_datetime(
    train_raw["pickup_scheduled_at"],
    errors="coerce"
)

customers["signup_date"] = pd.to_datetime(
    customers["signup_date"],
    errors="coerce"
)

products["launch_date"] = pd.to_datetime(
    products["launch_date"],
    errors="coerce"
)


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Kestrel Home Return Risk API",
    description="Pre-dispatch return-risk scoring API",
    version="1.0.0",
)


# ============================================================
# REQUEST MODEL
# ============================================================

class PredictionRequest(BaseModel):

    order_id: str
    order_placed_at: str

    customer_id: str
    sku: str

    sales_channel: str
    payment_mode: str

    discount_pct: float = Field(ge=0, le=100)
    qty: int = Field(ge=1)

    order_value_inr: float = Field(gt=0)

    promised_delivery_days: int = Field(ge=0)

    delivery_pincode: str

    is_gift: str

    customer_prior_orders: int = Field(ge=0)
    customer_prior_returns: int = Field(ge=0)

    delivery_note: Optional[str] = ""

    source: str


# ============================================================
# PAYMENT ANOMALY CORRECTION
# ============================================================

def correct_payment_anomaly(
    order_value,
    list_price,
    qty,
    discount_pct,
    order_date,
):

    expected_value = (
        list_price
        * qty
        * (1 - discount_pct / 100.0)
    )

    if expected_value <= 0:
        return float(order_value)

    ratio = order_value / expected_value

    if (
        order_date.year == 2025
        and order_date.month == 10
        and 95 <= ratio <= 105
    ):
        return float(expected_value)

    return float(order_value)


# ============================================================
# CUSTOMER HISTORY
# ============================================================

def get_customer_history(
    customer_id,
    order_date,
):

    history = train_raw[
        (
            train_raw["customer_id"].astype(str)
            == str(customer_id)
        )
        &
        (
            train_raw["order_placed_at"]
            < order_date
        )
    ].copy()

    if history.empty:
        return 0, 0

    prior_orders = len(history)

    prior_returns = int(
        history["returned"].sum()
    )

    return prior_orders, prior_returns


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def build_feature_row(request):

    order_date = pd.to_datetime(
        request.order_placed_at,
        errors="coerce"
    )

    if pd.isna(order_date):

        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid order_placed_at. "
                "Use format YYYY-MM-DDTHH:MM:SS"
            ),
        )

    # --------------------------------------------------------
    # CUSTOMER
    # --------------------------------------------------------

    customer_match = customers[
        customers["customer_id"].astype(str)
        == str(request.customer_id)
    ]

    if customer_match.empty:

        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown customer_id: "
                f"{request.customer_id}"
            ),
        )

    customer = customer_match.iloc[0]

    # --------------------------------------------------------
    # PRODUCT
    # --------------------------------------------------------

    product_match = products[
        products["sku"].astype(str)
        == str(request.sku)
    ]

    if product_match.empty:

        raise HTTPException(
            status_code=400,
            detail=f"Unknown SKU: {request.sku}",
        )

    product = product_match.iloc[0]

    # --------------------------------------------------------
    # CUSTOMER HISTORY
    # --------------------------------------------------------

    temporal_orders, temporal_returns = (
        get_customer_history(
            request.customer_id,
            order_date,
        )
    )

    temporal_return_rate = (
        temporal_returns / temporal_orders
        if temporal_orders > 0
        else 0.0
    )

    supplied_return_rate = (
        request.customer_prior_returns
        / request.customer_prior_orders
        if request.customer_prior_orders > 0
        else 0.0
    )

    # --------------------------------------------------------
    # PAYMENT
    # --------------------------------------------------------

    normalized_order_value = (
        correct_payment_anomaly(
            request.order_value_inr,
            float(product["list_price_inr"]),
            request.qty,
            request.discount_pct,
            order_date,
        )
    )

    # --------------------------------------------------------
    # DELIVERY NOTE
    # --------------------------------------------------------

    delivery_note = request.delivery_note or ""

    delivery_note_missing = int(
        len(delivery_note.strip()) == 0
    )

    delivery_note_length = len(delivery_note)

    delivery_note_word_count = len(
        delivery_note.split()
    )

    # --------------------------------------------------------
    # DATES
    # --------------------------------------------------------

    signup_date = pd.to_datetime(
        customer["signup_date"],
        errors="coerce",
    )

    launch_date = pd.to_datetime(
        product["launch_date"],
        errors="coerce",
    )

    customer_tenure_days = (
        (order_date - signup_date).days
        if not pd.isna(signup_date)
        else 0
    )

    product_age_days = (
        (order_date - launch_date).days
        if not pd.isna(launch_date)
        else 0
    )

    # --------------------------------------------------------
    # CALENDAR
    # --------------------------------------------------------

    order_year = order_date.year
    order_month = order_date.month
    order_day = order_date.day
    order_day_of_week = order_date.dayofweek

    order_week_of_year = int(
        order_date.isocalendar().week
    )

    order_hour = order_date.hour

    is_weekend = int(
        order_day_of_week >= 5
    )

    month_sin = np.sin(
        2 * np.pi * order_month / 12
    )

    month_cos = np.cos(
        2 * np.pi * order_month / 12
    )

    day_of_week_sin = np.sin(
        2 * np.pi * order_day_of_week / 7
    )

    day_of_week_cos = np.cos(
        2 * np.pi * order_day_of_week / 7
    )

    # --------------------------------------------------------
    # ORDER
    # --------------------------------------------------------

    order_value_per_unit = (
        normalized_order_value / request.qty
    )

    has_prior_return = int(
        request.customer_prior_returns > 0
    )

    multi_quantity_order = int(
        request.qty > 1
    )

    is_default_pincode = int(
        str(request.delivery_pincode).strip()
        == "000000"
    )

    has_customer_history = int(
        request.customer_prior_orders > 0
    )

    # --------------------------------------------------------
    # MODEL FEATURES
    # --------------------------------------------------------

    row = {

        # Numerical
        "discount_pct": request.discount_pct,
        "qty": request.qty,
        "order_value_inr": normalized_order_value,
        "promised_delivery_days":
            request.promised_delivery_days,

        "customer_prior_orders":
            request.customer_prior_orders,

        "customer_prior_returns":
            request.customer_prior_returns,

        "customer_prior_returns_temporal":
            temporal_returns,

        "customer_prior_return_rate_temporal":
            temporal_return_rate,

        "has_customer_history":
            has_customer_history,

        "customer_tenure_days":
            customer_tenure_days,

        "warranty_months":
            float(product["warranty_months"]),

        "product_age_days":
            product_age_days,

        "order_year":
            order_year,

        "order_month":
            order_month,

        "order_day":
            order_day,

        "order_day_of_week":
            order_day_of_week,

        "order_week_of_year":
            order_week_of_year,

        "order_hour":
            order_hour,

        "is_weekend":
            is_weekend,

        "month_sin":
            month_sin,

        "month_cos":
            month_cos,

        "day_of_week_sin":
            day_of_week_sin,

        "day_of_week_cos":
            day_of_week_cos,

        "is_default_pincode":
            is_default_pincode,

        "delivery_note_missing":
            delivery_note_missing,

        "delivery_note_length":
            delivery_note_length,

        "delivery_note_word_count":
            delivery_note_word_count,

        "order_value_per_unit":
            order_value_per_unit,

        "has_prior_return":
            has_prior_return,

        "multi_quantity_order":
            multi_quantity_order,

        "customer_prior_return_rate":
            supplied_return_rate,

        "list_price_inr":
            float(product["list_price_inr"]),

        # Categorical
        "sku":
            request.sku,

        "sales_channel":
            request.sales_channel,

        "payment_mode":
            request.payment_mode,

        "is_gift":
            request.is_gift,

        "source":
            request.source,

        "city":
            str(customer["city"]),

        "state":
            str(customer["state"]),

        "shield_member":
            str(customer["shield_member"]),

        "family":
            str(product["family"]),

        "model_name":
            str(product["model_name"]),
    }

    return pd.DataFrame([row])


# ============================================================
# RISK BAND
# ============================================================

def get_risk_band(score):

    if score < 0.10:
        return "VERY LOW"

    if score < 0.20:
        return "LOW"

    if score < 0.30:
        return "MEDIUM"

    if score < 0.40:
        return "HIGH"

    return "VERY HIGH"


# ============================================================
# EXPLANATION
# ============================================================

def explain_prediction(
    feature_row,
    transformed,
):

    feature_names = (
        preprocessor
        .get_feature_names_out()
    )

    reasons = []

    try:

        booster = model.get_booster()

        contributions = booster.predict(
            xgb.DMatrix(transformed),
            pred_contribs=True,
        )[0]

        contribution_map = {
            name: float(value)
            for name, value in zip(
                feature_names,
                contributions[:-1],
            )
        }

        positive = sorted(
            contribution_map.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        for feature_name, contribution in positive:

            if contribution <= 0:
                break

            readable = None

            if (
                "customer_prior_returns"
                in feature_name
            ):
                readable = (
                    "Customer has prior return history"
                )

            elif (
                "customer_prior_return_rate"
                in feature_name
            ):
                readable = (
                    "Customer has elevated "
                    "historical return rate"
                )

            elif (
                "payment_mode_cod"
                in feature_name
            ):
                readable = (
                    "COD payment mode"
                )

            elif (
                "family_Robot Vacuum"
                in feature_name
            ):
                readable = (
                    "Robot Vacuum product family"
                )

            elif (
                "family_Mixer Grinder"
                in feature_name
            ):
                readable = (
                    "Mixer Grinder product family"
                )

            elif (
                "sales_channel_marketplace"
                in feature_name
            ):
                readable = (
                    "Marketplace sales channel"
                )

            elif (
                "sales_channel_partner_outlet"
                in feature_name
            ):
                readable = (
                    "Partner-outlet sales channel"
                )

            elif (
                "is_gift_Y"
                in feature_name
            ):
                readable = "Gift order"

            elif (
                "promised_delivery_days"
                in feature_name
            ):
                readable = (
                    "Longer promised delivery window"
                )

            elif (
                "list_price_inr"
                in feature_name
            ):
                readable = (
                    "Higher product list price"
                )

            elif (
                "warranty_months"
                in feature_name
            ):
                readable = (
                    "Product warranty profile"
                )

            elif (
                "is_default_pincode"
                in feature_name
            ):
                readable = (
                    "Default / missing-address pincode"
                )

            if (
                readable
                and readable not in reasons
            ):
                reasons.append(readable)

            if len(reasons) >= 3:
                break

    except Exception:

        pass

    # --------------------------------------------------------
    # FALLBACKS
    # --------------------------------------------------------

    if not reasons:

        if (
            int(
                feature_row[
                    "customer_prior_returns"
                ].iloc[0]
            )
            > 0
        ):
            reasons.append(
                "Customer has prior return history"
            )

        if (
            feature_row[
                "payment_mode"
            ].iloc[0]
            == "cod"
        ):
            reasons.append(
                "COD payment mode"
            )

        if (
            feature_row[
                "family"
            ].iloc[0]
            == "Robot Vacuum"
        ):
            reasons.append(
                "Robot Vacuum product family"
            )

    if not reasons:

        reasons.append(
            "No dominant positive risk signal identified"
        )

    return reasons[:3]


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "ok",
        "model": "xgboost_return_risk_final",
        "model_loaded": True,
    }


# ============================================================
# PREDICT
# ============================================================

@app.post("/predict")
def predict(request: PredictionRequest):

    try:

        feature_row = build_feature_row(request)

        transformed = preprocessor.transform(
            feature_row
        )

        score = float(
            model.predict_proba(
                transformed
            )[0, 1]
        )

        band = get_risk_band(score)

        reasons = explain_prediction(
            feature_row,
            transformed,
        )

        intervention_threshold = 0.10

        return {
            "order_id":
                request.order_id,

            "score":
                round(score, 6),

            "risk_band":
                band,

            "intervention_recommended":
                bool(
                    score >= intervention_threshold
                ),

            "intervention_threshold":
                intervention_threshold,

            "reasons":
                reasons,

            "note":
                (
                    "Score is primarily a ranking "
                    "signal, not a calibrated probability."
                ),
        }

    except HTTPException:
        raise

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "Prediction failed: "
                f"{type(exc).__name__}: {exc}"
            ),
        )


# ============================================================
# DEMO ORDER
# ============================================================

@app.get("/demo-order")
def demo_order():

    if test_raw.empty:

        raise HTTPException(
            status_code=404,
            detail="No test data available.",
        )

    row = test_raw.iloc[0]

    result = {}

    for column in test_raw.columns:

        value = row[column]

        # Convert NumPy/Pandas values into
        # normal JSON-compatible Python values.

        if pd.isna(value):

            result[column] = ""

        elif isinstance(
            value,
            (
                np.integer,
                np.int64,
                np.int32,
            ),
        ):

            result[column] = int(value)

        elif isinstance(
            value,
            (
                np.floating,
                np.float64,
                np.float32,
            ),
        ):

            result[column] = float(value)

        elif isinstance(
            value,
            (
                pd.Timestamp,
            ),
        ):

            result[column] = (
                value.isoformat()
            )

        else:

            result[column] = str(value)

    return result


# ============================================================
# UI
# ============================================================

HTML_PAGE = """
<!DOCTYPE html>

<html>

<head>

<meta charset="UTF-8">

<meta name="viewport"
content="width=device-width, initial-scale=1.0">

<title>Kestrel Home Return Risk</title>

<style>

body {
    font-family: Arial, sans-serif;
    max-width: 1100px;
    margin: 40px auto;
    padding: 0 20px;
    background: #f5f6f8;
}

h1 {
    margin-bottom: 5px;
}

.subtitle {
    color: #666;
    margin-bottom: 25px;
}

.card {
    background: white;
    padding: 25px;
    border-radius: 12px;
    margin-bottom: 20px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.08);
}

.grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 15px;
}

label {
    display: block;
    font-weight: 600;
    margin-bottom: 5px;
}

input,
select,
textarea {
    width: 100%;
    padding: 10px;
    border: 1px solid #ccc;
    border-radius: 6px;
    box-sizing: border-box;
}

textarea {
    min-height: 80px;
}

button {
    padding: 12px 20px;
    border: none;
    border-radius: 7px;
    cursor: pointer;
    font-weight: bold;
    margin-right: 10px;
}

.predict {
    background: #111;
    color: white;
}

.demo {
    background: #ddd;
}

.result {
    display: none;
}

.score {
    font-size: 42px;
    font-weight: bold;
}

.band {
    display: inline-block;
    padding: 8px 14px;
    border-radius: 20px;
    background: #eee;
    font-weight: bold;
}

.reason {
    padding: 10px;
    margin: 7px 0;
    background: #f1f1f1;
    border-radius: 6px;
}

.error {
    color: #b00020;
    white-space: pre-wrap;
}

.note {
    color: #666;
    font-size: 13px;
    margin-top: 15px;
}

@media(max-width: 700px) {

    .grid {
        grid-template-columns: 1fr;
    }

}

</style>

</head>


<body>

<h1>Kestrel Home — Return Risk</h1>

<div class="subtitle">
Pre-dispatch return-risk scoring and intervention demo
</div>


<div class="card">

<h2>Order</h2>

<div class="grid">


<div>
<label>Order ID</label>
<input id="order_id">
</div>


<div>
<label>Order placed at</label>
<input id="order_placed_at">
</div>


<div>
<label>Customer ID</label>
<input id="customer_id">
</div>


<div>
<label>SKU</label>
<input id="sku">
</div>


<div>
<label>Sales channel</label>

<select id="sales_channel">

<option>app</option>
<option>web</option>
<option>marketplace</option>
<option>partner_outlet</option>

</select>

</div>


<div>
<label>Payment mode</label>

<select id="payment_mode">

<option>prepaid_upi</option>
<option>prepaid_card</option>
<option>cod</option>
<option>emi</option>

</select>

</div>


<div>
<label>Discount %</label>
<input id="discount_pct" type="number">
</div>


<div>
<label>Quantity</label>
<input id="qty" type="number">
</div>


<div>
<label>Order value ₹</label>
<input id="order_value_inr" type="number">
</div>


<div>
<label>Promised delivery days</label>
<input id="promised_delivery_days" type="number">
</div>


<div>
<label>Delivery pincode</label>
<input id="delivery_pincode">
</div>


<div>
<label>Gift order</label>

<select id="is_gift">
<option>N</option>
<option>Y</option>
</select>

</div>


<div>
<label>Prior orders</label>
<input id="customer_prior_orders" type="number">
</div>


<div>
<label>Prior returns</label>
<input id="customer_prior_returns" type="number">
</div>


<div>
<label>Source</label>

<select id="source">
<option>crm</option>
<option>partner_feed</option>
</select>

</div>

</div>


<br>


<label>Delivery note</label>

<textarea id="delivery_note"></textarea>


<br><br>


<button
class="predict"
onclick="predict()">

Predict return risk

</button>


<button
class="demo"
onclick="loadDemo()">

Load demo order

</button>

</div>


<div class="card result"
id="result">

<h2>Prediction</h2>


<div class="score"
id="score">

-

</div>


<div class="band"
id="band">

-

</div>


<p id="intervention"></p>


<h3>Reason signals</h3>

<div id="reasons"></div>


<div class="note"
id="note"></div>

</div>


<script>


function getValue(id) {

    return document
        .getElementById(id)
        .value;

}


async function loadDemo() {

    try {

        const response =
            await fetch("/demo-order");

        const data =
            await response.json();

        if (!response.ok) {

            throw new Error(
                data.detail || "Failed to load demo"
            );

        }


        document.getElementById("order_id").value =
            data.order_id || "";


        document.getElementById("order_placed_at").value =
            data.order_placed_at || "";


        document.getElementById("customer_id").value =
            data.customer_id || "";


        document.getElementById("sku").value =
            data.sku || "";


        document.getElementById("sales_channel").value =
            data.sales_channel || "app";


        document.getElementById("payment_mode").value =
            data.payment_mode || "prepaid_upi";


        document.getElementById("discount_pct").value =
            data.discount_pct || 0;


        document.getElementById("qty").value =
            data.qty || 1;


        document.getElementById("order_value_inr").value =
            data.order_value_inr || "";


        document.getElementById("promised_delivery_days").value =
            data.promised_delivery_days || 5;


        document.getElementById("delivery_pincode").value =
            data.delivery_pincode || "";


        document.getElementById("is_gift").value =
            data.is_gift || "N";


        document.getElementById("customer_prior_orders").value =
            data.customer_prior_orders || 0;


        document.getElementById("customer_prior_returns").value =
            data.customer_prior_returns || 0;


        document.getElementById("delivery_note").value =
            data.delivery_note || "";


        document.getElementById("source").value =
            data.source || "crm";


    }

    catch(error) {

        alert(
            "Could not load demo order: "
            + error.message
        );

    }

}


async function predict() {

    const payload = {

        order_id:
            getValue("order_id"),

        order_placed_at:
            getValue("order_placed_at"),

        customer_id:
            getValue("customer_id"),

        sku:
            getValue("sku"),

        sales_channel:
            getValue("sales_channel"),

        payment_mode:
            getValue("payment_mode"),

        discount_pct:
            Number(
                getValue("discount_pct")
            ),

        qty:
            Number(
                getValue("qty")
            ),

        order_value_inr:
            Number(
                getValue("order_value_inr")
            ),

        promised_delivery_days:
            Number(
                getValue(
                    "promised_delivery_days"
                )
            ),

        delivery_pincode:
            getValue("delivery_pincode"),

        is_gift:
            getValue("is_gift"),

        customer_prior_orders:
            Number(
                getValue(
                    "customer_prior_orders"
                )
            ),

        customer_prior_returns:
            Number(
                getValue(
                    "customer_prior_returns"
                )
            ),

        delivery_note:
            getValue("delivery_note"),

        source:
            getValue("source")

    };


    document.getElementById(
        "result"
    ).style.display = "block";


    document.getElementById(
        "reasons"
    ).innerHTML = "Scoring...";


    try {

        const response =
            await fetch(
                "/predict",
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body:
                        JSON.stringify(payload)
                }
            );


        const data =
            await response.json();


        if (!response.ok) {

            throw new Error(
                data.detail
                || "Prediction failed"
            );

        }


        document.getElementById(
            "score"
        ).innerText =
            Number(
                data.score
            ).toFixed(4);


        document.getElementById(
            "band"
        ).innerText =
            data.risk_band;


        document.getElementById(
            "intervention"
        ).innerText =
            data.intervention_recommended
            ? "Initial pilot intervention: RECOMMENDED"
            : "Initial pilot intervention: NOT RECOMMENDED";


        const reasons =
            document.getElementById(
                "reasons"
            );

        reasons.innerHTML = "";


        data.reasons.forEach(
            function(reason) {

                const div =
                    document.createElement(
                        "div"
                    );

                div.className =
                    "reason";

                div.innerText =
                    reason;

                reasons.appendChild(
                    div
                );

            }
        );


        document.getElementById(
            "note"
        ).innerText =
            data.note;

    }


    catch(error) {

        document.getElementById(
            "reasons"
        ).innerHTML =
            '<div class="error">'
            + error.message
            + '</div>';

    }

}

</script>


</body>

</html>
"""


# ============================================================
# HOME
# ============================================================

@app.get(
    "/",
    response_class=HTMLResponse
)
def home():

    return HTML_PAGE