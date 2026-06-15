# Chest X-ray Classifier - Streamlit app
# Run with:  streamlit run app.py
# Loads the model trained by the notebook and lets you upload an X-ray to get
# per-condition predictions plus a Grad-CAM heat map of where the model looked.
#
# Two modes (radio in the sidebar):
#   * Classify    - upload/pick an X-ray, see predictions + Grad-CAM.
#   * Monitoring  - population-level view built from the prediction log the app
#                   writes on every classification (the lightweight, self-contained
#                   version of a Power BI / clinical-analytics dashboard).
import os, io, json, csv, hashlib, datetime, random
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

HERE = os.path.dirname(os.path.abspath(__file__))
IMG_SIZE = (224, 224)
LOG_DIR = os.path.join(HERE, "logs")
PRED_LOG = os.path.join(LOG_DIR, "predictions_log.csv")
FEEDBACK_LOG = os.path.join(LOG_DIR, "feedback_log.csv")

st.set_page_config(page_title="Chest X-ray Classifier", layout="centered")

@st.cache_resource
def load_everything():
    # Saved model has two outputs: [conv feature map, predictions].
    model = keras.models.load_model(os.path.join(HERE, "cxr14_model.keras"))
    classes = json.load(open(os.path.join(HERE, "classes.json")))
    thresholds = np.array(json.load(open(os.path.join(HERE, "thresholds.json"))))
    return model, classes, thresholds

model, CLASSES, THRESHOLDS = load_everything()

# Ground-truth labels for the bundled sample images (for expected-vs-predicted display).
try:
    SAMPLE_LABELS = json.load(open(os.path.join(HERE, "sample_labels.json")))
except Exception:
    SAMPLE_LABELS = {}

def predict(arr):
    conv, preds = model(arr[None])
    return preds[0].numpy()

def gradcam(arr, class_idx):
    with tf.GradientTape() as tape:
        conv_out, preds = model(arr[None])
        loss = preds[:, class_idx]
    grads = tape.gradient(loss, conv_out)
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    cam = tf.reduce_sum(conv_out[0] * weights, axis=-1)
    cam = tf.maximum(cam, 0) / (tf.reduce_max(cam) + 1e-8)
    return cam.numpy()

# ---- Lightweight logging ----------------------------------------------------
# Every classification appends one row to predictions_log.csv. This is what turns
# a single-image demo into something you can monitor at the population level, and
# it is the data source the Monitoring tab (and any external BI tool) reads from.
def log_prediction(source, probs, flagged):
    os.makedirs(LOG_DIR, exist_ok=True)
    new = not os.path.exists(PRED_LOG)
    with open(PRED_LOG, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["timestamp", "source", "top_prediction", "top_probability",
                        "num_flagged"] + CLASSES)
        top = int(np.argmax(probs))
        w.writerow([datetime.datetime.now().isoformat(timespec="seconds"), source,
                    CLASSES[top], round(float(probs[top]), 4), len(flagged)]
                   + [round(float(p), 4) for p in probs])

def log_feedback(source, verdict, top_prediction):
    os.makedirs(LOG_DIR, exist_ok=True)
    new = not os.path.exists(FEEDBACK_LOG)
    with open(FEEDBACK_LOG, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["timestamp", "source", "top_prediction", "feedback"])
        w.writerow([datetime.datetime.now().isoformat(timespec="seconds"),
                    source, top_prediction, verdict])

# ---- Mode switch ------------------------------------------------------------
mode = st.sidebar.radio("Mode", ["Classify", "Monitoring"], index=0)
st.sidebar.caption("Classify runs the model on one X-ray. Monitoring summarises "
                   "everything classified so far from the prediction log.")

# =============================================================================
# MONITORING TAB
# =============================================================================
def monitoring_view():
    import pandas as pd
    st.title("Monitoring dashboard")
    st.write("Population-level view of every X-ray this app has classified. "
             "Built from `logs/predictions_log.csv`, which the app appends to on "
             "each prediction - the same data could feed Power BI or any BI tool.")
    if not os.path.exists(PRED_LOG):
        st.info("No predictions logged yet. Switch to **Classify**, run a few "
                "X-rays, then come back.")
        return
    df = pd.read_csv(PRED_LOG)
    if df.empty:
        st.info("The prediction log is empty. Classify a few X-rays first.")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("X-rays classified", len(df))
    c2.metric("Avg findings flagged / image", f"{df['num_flagged'].mean():.2f}")
    c3.metric("Avg top-prediction confidence", f"{df['top_probability'].mean():.2f}")

    st.subheader("How often each condition is flagged")
    flag_counts = {c: int((df[c] >= THRESHOLDS[i]).sum()) for i, c in enumerate(CLASSES)}
    fc = pd.Series(flag_counts).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(fc.index, fc.values, color="#10495C")
    ax.set_xlabel("number of images flagged (prob >= tuned threshold)")
    fig.tight_layout()
    st.pyplot(fig)

    st.subheader("Confidence distribution (top prediction per image)")
    fig2, ax2 = plt.subplots(figsize=(7, 3))
    ax2.hist(df["top_probability"], bins=20, color="#E07A33", edgecolor="white")
    ax2.set_xlabel("top-prediction probability"); ax2.set_ylabel("images")
    fig2.tight_layout()
    st.pyplot(fig2)

    st.subheader("Per-condition summary")
    per = pd.DataFrame({
        "Condition": CLASSES,
        "Times flagged": [int((df[c] >= THRESHOLDS[i]).sum()) for i, c in enumerate(CLASSES)],
        "Flagged %": [round(float((df[c] >= THRESHOLDS[i]).mean()) * 100, 1)
                      for i, c in enumerate(CLASSES)],
        "Avg probability": [round(float(df[c].mean()), 3) for c in CLASSES],
    }).sort_values("Times flagged", ascending=False)
    st.dataframe(per, use_container_width=True, hide_index=True)

    if os.path.exists(FEEDBACK_LOG):
        fb = pd.read_csv(FEEDBACK_LOG)
        if not fb.empty:
            st.subheader("User feedback")
            counts = fb["feedback"].value_counts().to_dict()
            st.write(f"Looks right: **{counts.get('looks_right', 0)}**  |  "
                     f"Looks off: **{counts.get('looks_off', 0)}**")

    st.subheader("Most recent classifications")
    st.dataframe(df.tail(15)[["timestamp", "source", "top_prediction",
                              "top_probability", "num_flagged"]],
                 use_container_width=True, hide_index=True)

# =============================================================================
# CLASSIFY TAB
# =============================================================================
def classify_view():
    st.title("Chest X-ray Multi-Label Classifier")
    st.write(
        "Upload a frontal chest X-ray. The model estimates the probability of 14 thoracic "
        "conditions (an image can have several at once) and highlights the region that drove "
        "the top prediction with Grad-CAM."
    )
    st.warning("Educational demo only - NOT a medical device. Do not use for real diagnosis.")

    # Pick an input: an upload, or one of the bundled sample X-rays.
    sample_dir = os.path.join(HERE, "samples")
    samples = sorted([f for f in os.listdir(sample_dir)]) if os.path.isdir(sample_dir) else []
    up = st.file_uploader("Upload a chest X-ray (PNG/JPG)", type=["png", "jpg", "jpeg"])
    choice = None
    if up is None and samples:
        # A random-sample button makes the live demo quick: one click loads a fresh case.
        if st.button("Random sample"):
            st.session_state["sample_select"] = random.choice(samples)
        choice = st.selectbox("...or try a bundled sample image", ["(none)"] + samples,
                              key="sample_select")

    img, truth, source = None, None, None
    if up is not None:
        img = Image.open(up).convert("RGB").resize(IMG_SIZE)
        source = f"upload:{up.name}"
    elif choice and choice != "(none)":
        img = Image.open(os.path.join(sample_dir, choice)).convert("RGB").resize(IMG_SIZE)
        truth = SAMPLE_LABELS.get(choice)   # known ground truth for bundled samples
        source = f"sample:{choice}"

    if img is None:
        st.caption("Waiting for an image...")
        return

    arr = np.asarray(img, np.float32)   # raw 0-255 pixels, matching how the model was trained
    probs = predict(arr)
    order = np.argsort(probs)[::-1]
    flagged = [CLASSES[i] for i in range(len(CLASSES)) if probs[i] >= THRESHOLDS[i]]

    # Log once per distinct image (Streamlit re-runs the script on every widget
    # interaction; this guard stops the same prediction being logged repeatedly).
    key = hashlib.md5((source or "").encode() + arr.tobytes()).hexdigest()
    if st.session_state.get("last_logged_key") != key:
        log_prediction(source or "unknown", probs, flagged)
        st.session_state["last_logged_key"] = key
        st.session_state.pop("feedback_done", None)

    st.subheader("Predicted conditions")
    rows = [{"Condition": CLASSES[i],
             "Probability": round(float(probs[i]), 3),
             "Flagged": "YES" if probs[i] >= THRESHOLDS[i] else ""} for i in order]
    st.dataframe(rows, use_container_width=True, hide_index=True)
    st.info("Flagged (above tuned threshold): " + (", ".join(flagged) if flagged else "none"))

    # Let the user export this prediction (one row per condition).
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["Condition", "Probability", "Threshold", "Flagged"])
    for i in order:
        w.writerow([CLASSES[i], round(float(probs[i]), 4), round(float(THRESHOLDS[i]), 2),
                    "YES" if probs[i] >= THRESHOLDS[i] else ""])
    st.download_button("Download predictions (CSV)", buf.getvalue(),
                       file_name="prediction.csv", mime="text/csv")

    # For bundled samples we know the real answer, so show expected vs predicted.
    if truth is not None:
        truth_set = {t for t in truth.split("|") if t and t != "No Finding"}
        st.success("Ground truth for this sample: " + (truth if truth else "No Finding"))
        if truth_set:
            hits = sorted(truth_set & set(flagged))
            st.write(f"Caught {len(hits)} of {len(truth_set)} true finding(s): "
                     + (", ".join(hits) if hits else "none"))

    top = int(order[0])
    cam = gradcam(arr, top)
    cam_big = zoom(cam, (IMG_SIZE[0] / cam.shape[0], IMG_SIZE[1] / cam.shape[1]), order=1)
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].imshow(img); ax[0].set_title("X-ray"); ax[0].axis("off")
    ax[1].imshow(img); ax[1].imshow(cam_big, cmap="jet", alpha=0.4)
    ax[1].set_title("Grad-CAM: " + CLASSES[top]); ax[1].axis("off")
    st.pyplot(fig)

    # Simple human-in-the-loop feedback, logged for the Monitoring tab.
    st.divider()
    st.caption("Does this prediction look reasonable? (logged for monitoring)")
    fcol1, fcol2, _ = st.columns([1, 1, 3])
    if not st.session_state.get("feedback_done"):
        if fcol1.button("Looks right"):
            log_feedback(source or "unknown", "looks_right", CLASSES[top])
            st.session_state["feedback_done"] = True
            st.toast("Thanks - feedback logged.")
        if fcol2.button("Looks off"):
            log_feedback(source or "unknown", "looks_off", CLASSES[top])
            st.session_state["feedback_done"] = True
            st.toast("Thanks - feedback logged.")
    else:
        st.caption("Feedback recorded for this image.")

# ---- Route ------------------------------------------------------------------
if mode == "Monitoring":
    monitoring_view()
else:
    classify_view()
