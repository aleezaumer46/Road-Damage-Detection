import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Smart Road Damage Detection",
    page_icon="🛣️",
    layout="wide"
)

# ==========================================
# CUSTOM CSS
# ==========================================
st.markdown("""
<style>

.main-title {
    text-align: center;
    font-size: 38px;
    font-weight: 700;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 17px;
    margin-bottom: 25px;
}

.result-card {
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #ddd;
    margin-bottom: 15px;
}

.info-box {
    padding: 15px;
    border-radius: 10px;
    border: 1px solid #ddd;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# LOAD MODEL
# ==========================================
@st.cache_resource
def load_model():
    return YOLO("best.pt")


model = load_model()

# ==========================================
# CLASS NAMES
# ==========================================
class_names = {
    "D00": "Longitudinal Crack",
    "D10": "Transverse Crack",
    "D20": "Alligator Crack",
    "D40": "Pothole",
    "D43": "Repair"
}

# ==========================================
# SEVERITY FUNCTION
# ==========================================
def calculate_severity(box, image_width, image_height, damage_type):

    x1, y1, x2, y2 = box.xyxy[0].tolist()

    box_width = x2 - x1
    box_height = y2 - y1

    damage_area = box_width * box_height
    image_area = image_width * image_height

    area_percentage = (damage_area / image_area) * 100

    if damage_type in ["Pothole", "Repair"]:

        if area_percentage >= 15:
            severity = "High"
        elif area_percentage >= 5:
            severity = "Medium"
        else:
            severity = "Low"

    else:

        if area_percentage >= 20:
            severity = "High"
        elif area_percentage >= 8:
            severity = "Medium"
        else:
            severity = "Low"

    return severity


# ==========================================
# MAINTENANCE RECOMMENDATION
# ==========================================
def get_recommendation(severity):

    if severity == "Low":
        return "No Immediate Action Required"

    elif severity == "Medium":
        return "Schedule Maintenance"

    else:
        return "Immediate Repair Required"


# ==========================================
# HEADER
# ==========================================
st.markdown(
    '<div class="main-title">🛣️ Smart Road Damage Detection</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered road damage detection using YOLO'
    '</div>',
    unsafe_allow_html=True
)

st.divider()

# ==========================================
# SIDEBAR
# ==========================================
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

    st.divider()

    st.subheader("📊 Output")

    st.write("• Damage Type")
    st.write("• Confidence Score")
    st.write("• Severity Level")
    st.write("• Maintenance Recommendation")

    st.divider()

    st.caption("Model: YOLO-based Road Damage Detector")
    st.caption("Input: JPG / JPEG / PNG")


# ==========================================
# MAIN CONTENT
# ==========================================
st.subheader("📤 Upload Road Image")

st.write(
    "Upload a clear road image to detect possible "
    "road damage."
)

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

# ==========================================
# PROCESS IMAGE
# ==========================================
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    image_width, image_height = image.size

    # --------------------------------------
    # Display uploaded image
    # --------------------------------------
    st.subheader("🖼️ Uploaded Image")

    st.image(
        image,
        caption="Road Image",
        use_container_width=True
    )

    # --------------------------------------
    # Temporary file
    # --------------------------------------
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".jpg"
    ) as temp_file:

        image.save(temp_file.name)
        temp_path = temp_file.name

    # --------------------------------------
    # Prediction
    # --------------------------------------
    with st.spinner("🔍 Analyzing road image..."):

        results = model.predict(
            source=temp_path,
            conf=0.25
        )

    result = results[0]

    st.divider()

    # ======================================
    # RESULTS
    # ======================================
    st.subheader("📊 Detection Results")

    # ======================================
    # NO DAMAGE
    # ======================================
    if len(result.boxes) == 0:

        st.success("✅ No road damage detected.")

        st.info(
            "Maintenance Recommendation: "
            "No Immediate Action Required"
        )

    # ======================================
    # DAMAGE DETECTED
    # ======================================
    else:

        # ----------------------------------
        # Annotated image
        # ----------------------------------
        annotated_image = result.plot()

        st.subheader("🎯 Detected Damage")

        st.image(
            annotated_image,
            caption="AI Detection Result",
            use_container_width=True
        )

        # ----------------------------------
        # Process detections
        # ----------------------------------
        for i, box in enumerate(result.boxes):

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            technical_name = model.names[class_id]

            damage_name = class_names.get(
                technical_name,
                technical_name
            )

            severity = calculate_severity(
                box,
                image_width,
                image_height,
                damage_name
            )

            recommendation = get_recommendation(
                severity
            )

            # --------------------------------
            # Detection Heading
            # --------------------------------
            st.markdown(
                f"### 🔎 Detection {i + 1}"
            )

            # --------------------------------
            # Result Columns
            # --------------------------------
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Damage Type",
                    damage_name
                )

            with col2:
                st.metric(
                    "Confidence",
                    f"{confidence * 100:.2f}%"
                )

            with col3:
                st.metric(
                    "Severity",
                    severity
                )

            with col4:
                st.metric(
                    "Action",
                    recommendation
                )

            st.divider()

    # ======================================
    # CLEAN TEMP FILE
    # ======================================
    os.remove(temp_path)


# ==========================================
# FOOTER
# ==========================================
st.divider()

st.caption(
    "AI-Based Smart Road Damage Detection | "
    "YOLO Road Damage Model"
)