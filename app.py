import streamlit as st
from ultralytics import YOLO
from PIL import Image
import pandas as pd
import tempfile
import os

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Smart Road Damage Detection",
    page_icon="🛣️",
    layout="wide"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 42px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 20px;
        text-align: center;
        color: #666;
        margin-bottom: 25px;
    }

    .info-box {
        padding: 15px;
        border-radius: 10px;
        background-color: #f5f5f5;
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================================================
# TITLE
# =========================================================

st.markdown(
    '<div class="main-title">🛣️ Smart Road Damage Detection</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-powered road damage detection using YOLO</div>',
    unsafe_allow_html=True
)

st.write(
    "Upload a road image and the trained YOLO model will detect "
    "possible road damage."
)

# =========================================================
# MODEL PATH
# =========================================================

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best.pt"

# =========================================================
# DAMAGE CLASS NAMES
# =========================================================

damage_names = {
    0: "Longitudinal Crack",
    1: "Transverse Crack",
    2: "Alligator Crack",
    3: "Pothole",
    4: "Repair"
}

# =========================================================
# SEVERITY
# =========================================================

def get_severity(confidence):
    """
    Confidence-based severity indicator.
    Note: This represents model confidence, not physical damage severity.
    """

    if confidence >= 0.70:
        return "High"
    elif confidence >= 0.40:
        return "Medium"
    else:
        return "Low"


# =========================================================
# MAINTENANCE RECOMMENDATION
# =========================================================

def get_recommendation(severity):

    if severity == "High":
        return (
            "Priority inspection is recommended. "
            "Schedule maintenance as soon as possible."
        )

    elif severity == "Medium":
        return (
            "Schedule maintenance and monitor the damaged "
            "area for further deterioration."
        )

    else:
        return (
            "Monitor the area and perform maintenance "
            "if damage increases."
        )


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)


try:
    model = load_model()

except Exception as e:
    st.error("❌ Unable to load the YOLO model.")
    st.error(str(e))
    st.stop()


# =========================================================
# MODEL INFORMATION
# =========================================================

with st.expander("🤖 Model Information"):

    st.write("**Model:** YOLO")
    st.write("**Model File:** best.pt")
    st.write("**Image Size:** 640 × 640")
    st.write("**Detection Threshold:** 20%")

# =========================================================
# IMAGE UPLOAD
# =========================================================

st.subheader("📤 Upload Road Image")

uploaded_file = st.file_uploader(
    "Upload a clear road image to detect possible road damage.",
    type=["jpg", "jpeg", "png"]
)

# =========================================================
# PROCESS IMAGE
# =========================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    # =====================================================
    # UPLOADED IMAGE
    # =====================================================

    st.subheader("🖼️ Uploaded Image")

    st.image(
        image,
        caption="Road Image",
        use_container_width=True
    )

    # =====================================================
    # SAVE IMAGE TEMPORARILY
    # =====================================================

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".jpg"
    ) as temp_file:

        image.save(temp_file.name)
        temp_image_path = temp_file.name

    # =====================================================
    # RUN YOLO
    # =====================================================

    with st.spinner("🔍 Detecting road damage..."):

        results = model.predict(
            source=temp_image_path,
            conf=0.20,
            imgsz=640,
            verbose=False
        )

    # =====================================================
    # REMOVE TEMPORARY IMAGE
    # =====================================================

    if os.path.exists(temp_image_path):
        os.remove(temp_image_path)

    result = results[0]

    # =====================================================
    # DETECTION VISUALIZATION
    # =====================================================

    st.subheader("🎯 Detection Visualization")

    annotated_image = result.plot()

    st.image(
        annotated_image,
        caption="Detected Road Damage",
        use_container_width=True
    )

    # =====================================================
    # DETECTION RESULTS
    # =====================================================

    st.subheader("📊 Detection Results")

    detections = []

    if result.boxes is not None and len(result.boxes) > 0:

        boxes = result.boxes

        for i in range(len(boxes)):

            class_id = int(boxes.cls[i].item())

            confidence = float(boxes.conf[i].item())

            # Damage name
            damage_type = damage_names.get(
                class_id,
                model.names.get(class_id, "Unknown")
            )

            # Confidence-based severity
            severity = get_severity(confidence)

            # Recommendation
            recommendation = get_recommendation(severity)

            detections.append(
                {
                    "Damage Type": damage_type,
                    "Confidence": confidence,
                    "Severity": severity,
                    "Maintenance Recommendation": recommendation
                }
            )

        # =================================================
        # NUMBER OF DETECTIONS
        # =================================================

        st.success(
            f"✅ {len(detections)} damage detection(s) found."
        )

        # =================================================
        # INDIVIDUAL DETECTIONS
        # =================================================

        for i, detection in enumerate(detections, start=1):

            st.markdown(
                f"### 🔎 Detection {i}"
            )

            st.markdown(
                f"**Damage Type:** {detection['Damage Type']}"
            )

            st.markdown(
                f"**Confidence:** "
                f"**{detection['Confidence'] * 100:.2f}%**"
            )

            st.markdown(
                f"**Severity Indicator:** {detection['Severity']}"
            )

            st.markdown(
                f"**Maintenance Recommendation:** "
                f"{detection['Maintenance Recommendation']}"
            )

            st.divider()

        # =================================================
        # DETECTION SUMMARY
        # =================================================

        st.subheader("📋 Detection Summary")

        df = pd.DataFrame(detections)

        # Format confidence for display
        display_df = df.copy()

        display_df["Confidence"] = (
            display_df["Confidence"] * 100
        ).round(2).astype(str) + "%"

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

        # =================================================
        # DAMAGE TYPE SUMMARY
        # =================================================

        st.subheader("📈 Damage Type Summary")

        damage_counts = (
            df["Damage Type"]
            .value_counts()
            .reset_index()
        )

        damage_counts.columns = [
            "Damage Type",
            "Number of Detections"
        ]

        st.dataframe(
            damage_counts,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.warning(
            "⚠️ No road damage was detected in this image."
        )

# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "AI-Based Smart Road Damage Detection | YOLO Road Damage Model"
)