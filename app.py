import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
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

st.markdown("""
<style>
    .main-title {
        text-align: center;
        font-size: 38px;
        font-weight: bold;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        margin-bottom: 30px;
    }

    .result-box {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #ddd;
        margin-bottom: 15px;
    }

    .confidence {
        font-size: 18px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

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
# DAMAGE CLASS INFORMATION
# =========================================================

CLASS_NAMES = {
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
    if confidence >= 0.75:
        return "High"
    elif confidence >= 0.50:
        return "Medium"
    else:
        return "Low"


# =========================================================
# MAINTENANCE RECOMMENDATION
# =========================================================

def get_recommendation(damage_type, severity):

    if severity == "High":
        return "Immediate inspection and maintenance required."

    elif severity == "Medium":
        return "Schedule maintenance and monitor the damaged area."

    else:
        return "Monitor the area and perform maintenance if damage increases."


# =========================================================
# LOAD MODEL
# =========================================================

MODEL_PATH = "best.pt"

if not os.path.exists(MODEL_PATH):
    st.error(
        f"Model file '{MODEL_PATH}' was not found. "
        "Please make sure best.pt is in the project folder."
    )
    st.stop()

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)


model = load_model()

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("📌 About the Application")

    st.write(
        "This application uses a trained YOLO model "
        "to detect different types of road damage."
    )

    st.subheader("🔍 Supported Damage Types")

    st.write("• Longitudinal Crack")
    st.write("• Transverse Crack")
    st.write("• Alligator Crack")
    st.write("• Pothole")
    st.write("• Repair")

    st.subheader("📊 Output")

    st.write("• Damage Type")
    st.write("• Confidence Score")
    st.write("• Severity Level")
    st.write("• Maintenance Recommendation")

    st.divider()

    st.write("Model: YOLO-based Road Damage Detector")
    st.write("Input: JPG / JPEG / PNG")

# =========================================================
# IMAGE UPLOAD
# =========================================================

st.header("📤 Upload Road Image")

uploaded_file = st.file_uploader(
    "Upload a clear road image to detect possible road damage.",
    type=["jpg", "jpeg", "png"]
)

# =========================================================
# PREDICTION
# =========================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("🖼️ Uploaded Image")

    st.image(
        image,
        caption="Road Image",
        use_container_width=True
    )

    # Convert image to numpy array
    image_array = np.array(image)

    # =====================================================
    # RUN YOLO
    # =====================================================

    with st.spinner("🔍 Detecting road damage..."):

        results = model.predict(
            source=image_array,
            imgsz=640,
            conf=0.25,
            verbose=False
        )

    result = results[0]

    # =====================================================
    # DISPLAY DETECTION IMAGE
    # =====================================================

    st.subheader("🎯 Detection Visualization")

    annotated_image = result.plot()

    annotated_image = annotated_image[:, :, ::-1]

    st.image(
        annotated_image,
        caption="Detected Road Damage",
        use_container_width=True
    )

    # =====================================================
    # DETECTION RESULTS
    # =====================================================

    st.subheader("📊 Detection Results")

    boxes = result.boxes

    if boxes is None or len(boxes) == 0:

        st.warning(
            "⚠️ No road damage was detected in this image."
        )

    else:

        st.success(
            f"✅ {len(boxes)} damage detection(s) found."
        )

        for i, box in enumerate(boxes):

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            damage_type = CLASS_NAMES.get(
                class_id,
                result.names.get(class_id, "Unknown")
            )

            severity = get_severity(confidence)

            recommendation = get_recommendation(
                damage_type,
                severity
            )

            st.markdown(
                f"""
                <div class="result-box">

                <h3>🔎 Detection {i + 1}</h3>

                <b>Damage Type:</b> {damage_type}<br><br>

                <b>Confidence:</b>
                <span class="confidence">
                {confidence * 100:.2f}%
                </span>
                <br><br>

                <b>Severity:</b> {severity}<br><br>

                <b>Maintenance Recommendation:</b>
                {recommendation}

                </div>
                """,
                unsafe_allow_html=True
            )

        # =================================================
        # SUMMARY TABLE
        # =================================================

        st.subheader("📋 Detection Summary")

        summary_data = []

        for i, box in enumerate(boxes):

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            damage_type = CLASS_NAMES.get(
                class_id,
                result.names.get(class_id, "Unknown")
            )

            severity = get_severity(confidence)

            summary_data.append(
                {
                    "Detection": i + 1,
                    "Damage Type": damage_type,
                    "Confidence": f"{confidence * 100:.2f}%",
                    "Severity": severity
                }
            )

        st.dataframe(
            summary_data,
            use_container_width=True,
            hide_index=True
        )

# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "AI-Based Smart Road Damage Detection | YOLO Road Damage Model"
)