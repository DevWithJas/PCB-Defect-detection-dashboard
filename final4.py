import streamlit as st
import os
import io
import zipfile
import torch
import torchvision
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from albumentations.pytorch import ToTensorV2
import random
import shutil
import tempfile
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av

# ---------------------------------
# Configuration & Global Variables
# ---------------------------------
st.set_page_config(
    page_title="🔧 High-Tech PCB Defect Detection Dashboard",
    page_icon="🤖",
    layout="wide",
)

# Custom CSS for Blue Navbar (Sidebar)
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        background-color: #152238;
        background-image: none;
    }
    [data-testid="stSidebar"] * {
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Add Custom Background
st.markdown(
        f"""
        <style>
        .stApp {{
            background: url("https://i.gifer.com/2RNb.gif");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Header Section with GIF
st.markdown(
    f"""
    <style>
    [data-testid="stAppViewContainer"] > .main {{
        background: url("https://i.gifer.com/2RNb.gif");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Class mapping (example)
CLASSES = {
    "missing_hole": 0,
    "mouse_bite": 1,
    "open_circuit": 2,
    "short": 3,
    "spur": 4,
    "spurious_copper": 5,
}
class_map_inv = {v: k for k, v in CLASSES.items()}

# Device Configuration
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Create a temporary directory for extracted images
if "temp_dir" not in st.session_state:
    st.session_state["temp_dir"] = tempfile.mkdtemp()

if "model_loaded" not in st.session_state:
    st.session_state["model_loaded"] = False
if "images_loaded" not in st.session_state:
    st.session_state["images_loaded"] = False
if "image_list" not in st.session_state:
    st.session_state["image_list"] = []
if "model_conf_threshold" not in st.session_state:
    st.session_state["model_conf_threshold"] = 0.3

# ---------------------------------
# Model Loading Utility
# ---------------------------------
@st.cache_resource
def get_base_model(num_classes=6):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.to(device)
    return model

model = get_base_model(num_classes=len(CLASSES))

# ---------------------------------
# Sidebar Navigation
# ---------------------------------
st.sidebar.title("🔧 Options Panel")
st.sidebar.markdown("**Navigate the high-tech dashboard using these options!** 🤖")

app_mode = st.sidebar.radio(
    "📂 Primary Steps",
    ["About Project 📜", "Upload Model 🧠", "Upload Dataset 🗂️", "Data Visualization 📊"]
)

# Advanced options in expander
with st.sidebar.expander("Advanced Options"):
    test_model_selected = st.checkbox("Test Model ⚡")
    real_time_detection_selected = st.checkbox("Real-Time Detection 🎥")

if test_model_selected:
    app_mode = "Test Model ⚡"
elif real_time_detection_selected:
    app_mode = "Real-Time Detection 🎥"

# ---------------------------------
# Helper Functions
# ---------------------------------
def load_image_as_tensor(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    image /= 255.0
    to_tensor = ToTensorV2(p=1.0)
    sample = to_tensor(image=image)
    img_tensor = sample['image']
    return img_tensor

def display_technical_images(image_paths, num_samples=5):
    """
    Display images in a more technical manner. For each image:
    - Show the image
    - Display a histogram of pixel intensities.
    """
    samples = random.sample(image_paths, min(num_samples, len(image_paths)))
    for img_path in samples:
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].imshow(img_rgb)
        axes[0].set_title(os.path.basename(img_path))
        axes[0].axis('off')

        # Flatten the image to get intensity values
        # Compute histogram for all channels (R,G,B)
        colors = ['r', 'g', 'b']
        for i, col in enumerate(colors):
            hist = cv2.calcHist([img], [i], None, [256], [0,256])
            axes[1].plot(hist, color=col)
        axes[1].set_title("Pixel Intensity Histogram")
        axes[1].set_xlim([0,256])

        st.pyplot(fig)

def run_inference_on_image_tensor(image_tensor, confidence_threshold=0.3):
    image_tensor = image_tensor.unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        prediction = model(image_tensor)[0]
    keep_indices = [i for i, score in enumerate(prediction['scores']) if score.item() >= confidence_threshold]
    filtered_boxes = prediction['boxes'][keep_indices].cpu().numpy()
    filtered_scores = prediction['scores'][keep_indices].cpu().numpy()
    filtered_labels = prediction['labels'][keep_indices].cpu().numpy()
    return filtered_boxes, filtered_scores, filtered_labels

def draw_boxes_on_frame(frame, boxes, scores, labels, threshold=0.3):
    for i, box in enumerate(boxes):
        x, y, x2, y2 = box
        cv2.rectangle(frame, (int(x), int(y)), (int(x2), int(y2)), (0, 255, 0), 2)
        label = class_map_inv[labels[i]] if labels[i] in class_map_inv else "Unknown"
        score = scores[i]
        if score >= threshold:
            cv2.putText(
                frame,
                f"{label}: {score:.2f}",
                (int(x), int(y) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )
    return frame

# ---------------------------------
# WebRTC for Real-Time Detection
# ---------------------------------
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    frame_tensor = ToTensorV2()(image=frame_rgb)['image']

    confidence_threshold = st.session_state["model_conf_threshold"]
    boxes, scores, labels = run_inference_on_image_tensor(frame_tensor, confidence_threshold=confidence_threshold)
    annotated_frame = draw_boxes_on_frame(img, boxes, scores, labels, threshold=confidence_threshold)

    return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

def real_time_detection():
    st.header("🎥 Real-Time PCB Defect Detection")
    if not st.session_state["model_loaded"]:
        st.warning("Please upload and load your model first.")
        return

    st.write("Activate your camera to test real-time defect detection!")
    confidence_threshold = st.slider(
        "Set Confidence Threshold", 0.0, 1.0, st.session_state["model_conf_threshold"], 0.01
    )
    st.session_state["model_conf_threshold"] = confidence_threshold

    webrtc_streamer(
        key="example",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
    )

# ---------------------------------
# Pages
# ---------------------------------
if app_mode == "About Project 📜":
    st.header("📜 About This Project")
    st.markdown("""
    <style>
    .about-text {
        font-size: 22px;
        line-height: 1.6;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="about-text">
    Welcome to the High-Tech PCB Defect Detection Dashboard!
    
    This application leverages a Faster R-CNN object detection model to identify defects on Printed Circuit Boards (PCBs). 
    By uploading your custom trained model and a dataset of PCB images, you can visualize, test, and run real-time detection.
    
    **Key Features:**
    - **Model Upload & Management:** Easily upload your pre-trained Faster R-CNN model.
    - **Dataset Handling:** Upload a zip of PCB images and visualize them.
    - **Data Visualization:** View images with technical histograms.
    - **Testing & Real-Time Detection:** Run inference on static images or in real-time using WebRTC.
    
    </div>
    """, unsafe_allow_html=True)


elif app_mode == "Upload Model 🧠":
    st.header("🧠 Upload Your Trained Faster R-CNN Model")
    uploaded_model = st.file_uploader("Upload model (.pt)", type=["pt"])
    if uploaded_model is not None:
        bytes_data = uploaded_model.read()
        if len(bytes_data) == 0:
            st.error("Uploaded file is empty.")
        else:
            try:
                buffer = io.BytesIO(bytes_data)
                state_dict = torch.load(buffer, map_location=device)
                model.load_state_dict(state_dict)
                model.eval()
                st.session_state["model_loaded"] = True
                st.success("🎉 Model loaded successfully!")
            except Exception as e:
                st.error(f"Error loading model: {e}")

    if st.session_state["model_loaded"]:
        st.info("🧩 Model is ready. Proceed to 'Upload Dataset'.")

elif app_mode == "Upload Dataset 🗂️":
    st.header("🗂️ Upload a Zip File of Images for Testing")
    st.write("📂 Upload a `.zip` file containing your test images.")

    uploaded_zip = st.file_uploader("Upload zip file")

    if uploaded_zip is not None:
        if not uploaded_zip.name.lower().endswith(".zip"):
            st.error("Please upload a file with .zip extension.")
        else:
            shutil.rmtree(st.session_state["temp_dir"])
            st.session_state["temp_dir"] = tempfile.mkdtemp()

            try:
                with zipfile.ZipFile(uploaded_zip) as zip_ref:
                    zip_ref.extractall(st.session_state["temp_dir"])
            except Exception as e:
                st.error(f"Error extracting zip: {e}")

            image_list = []
            for root, dirs, files in os.walk(st.session_state["temp_dir"]):
                for file in files:
                    if file.lower().endswith((".jpg", ".jpeg", ".png")):
                        image_list.append(os.path.join(root, file))

            if len(image_list) == 0:
                st.error("No images found in the uploaded zip. Please upload a valid dataset.")
            else:
                st.session_state["image_list"] = image_list
                st.session_state["images_loaded"] = True
                st.success(f"✅ Successfully loaded {len(image_list)} images!")

    if st.session_state["images_loaded"]:
        st.info("📊 Dataset is ready. Proceed to 'Data Visualization'.")

elif app_mode == "Data Visualization 📊":
    st.header("📊 Data Visualization")
    if not st.session_state["images_loaded"]:
        st.warning("Please upload your dataset first.")
    else:
        st.write("🔧 Displaying Technical Layout of Selected Images:")
        st.write("Here, you will see each image alongside a histogram of its pixel intensities.")
        display_technical_images(st.session_state["image_list"], num_samples=5)
        st.info("This technical visualization helps you inspect image quality and pixel distributions.")

elif app_mode == "Test Model ⚡":
    st.header("⚡ Test Your Model on the Uploaded Images")
    if not st.session_state["model_loaded"]:
        st.warning("Please upload and load your model first.")
    elif not st.session_state["images_loaded"]:
        st.warning("Please upload and load your dataset first.")
    else:
        name_to_path = {os.path.basename(p): p for p in st.session_state["image_list"]}
        selected_image_name = st.selectbox("Select an image to test:", list(name_to_path.keys()))
        confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, st.session_state["model_conf_threshold"], 0.01)

        if st.button("Run Prediction 🛠️"):
            img_path = name_to_path[selected_image_name]
            filtered_boxes, filtered_scores, filtered_labels = run_inference_on_image_tensor(
                load_image_as_tensor(img_path), confidence_threshold=confidence_threshold
            )

            img_np = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

            fig, ax = plt.subplots(figsize=(12, 7))
            ax.imshow(img_np)
            for i, box in enumerate(filtered_boxes):
                x, y, x2, y2 = box
                w, h = x2 - x, y2 - y
                rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='red', facecolor='none')
                ax.add_patch(rect)
                cls_name = class_map_inv[filtered_labels[i]] if filtered_labels[i] in class_map_inv else "Unknown"
                score = filtered_scores[i]
                ax.text(x, y - 10, f"{cls_name} ({score:.2f})", color='red', fontsize=10, weight='bold')

            st.pyplot(fig)
            st.info("🎯 Prediction complete!")

elif app_mode == "Real-Time Detection 🎥":
    real_time_detection()
