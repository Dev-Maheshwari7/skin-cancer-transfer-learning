
import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

st.write("STARTED")
# ── Config ────────────────────────────────────────────────────
LABEL_NAMES = {
    0: 'Actinic Keratoses (akiec)',
    1: 'Basal Cell Carcinoma (bcc)',
    2: 'Benign Keratosis (bkl)',
    3: 'Dermatofibroma (df)',
    4: 'Melanoma (mel)',
    5: 'Melanocytic Nevi (nv)',
    6: 'Vascular Lesion (vasc)'
}

RISK_LEVEL = {
    0: ('High',   '🔴'),
    1: ('High',   '🔴'),
    2: ('Low',    '🟢'),
    3: ('Low',    '🟢'),
    4: ('High',   '🔴'),
    5: ('Low',    '🟢'),
    6: ('Medium', '🟡')
}

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

SAMPLES = {
    'Sample 1 — Melanoma':             'samples/mel.jpg',
    'Sample 2 — Benign Nevi':          'samples/nv.jpg',
    'Sample 3 — Basal Cell Carcinoma': 'samples/bcc.jpg',
}

# ── Load model ────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 7)
    model.load_state_dict(torch.load('resnet50_skin_first.pth', map_location='cpu'))
    model.eval()
    return model

# ── Grad-CAM function ─────────────────────────────────────────
def run_gradcam(model, img_pil):
    target_layers = [model.layer4[-1]]

    # image for overlay — float32 [0,1]
    img_np     = np.array(img_pil.resize((224, 224))) / 255.0

    # image for model
    img_tensor = transform(img_pil).unsqueeze(0)

    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=img_tensor, targets=None)
        grayscale_cam = grayscale_cam[0]

    visualization = show_cam_on_image(
        img_np.astype(np.float32),
        grayscale_cam,
        use_rgb=True
    )
    return img_tensor, visualization

# ── UI ────────────────────────────────────────────────────────
st.title("🔬 Skin Lesion Classifier")
st.write("ResNet-50 fine-tuned on HAM10000 · 80% test accuracy · 10,015 dermatoscopic images")

model = load_model()

mode = st.radio("Choose input", ["Try a sample image", "Upload your own"])

img = None
if mode == "Try a sample image":
    choice = st.selectbox("Pick a sample", list(SAMPLES.keys()))
    img    = Image.open(SAMPLES[choice]).convert('RGB')
    st.image(img, caption=choice, width=300)
else:
    uploaded = st.file_uploader("Upload a skin lesion image", type=['jpg','jpeg','png'])
    if uploaded:
        img = Image.open(uploaded).convert('RGB')
        st.image(img, caption="Uploaded image", width=300)

# ── Predict + Grad-CAM ────────────────────────────────────────
if img and st.button("Analyse"):

    with st.spinner("Running model..."):
        img_tensor, gradcam_viz = run_gradcam(model, img)

        with torch.no_grad():
            output     = model(img_tensor)
            probs      = torch.softmax(output, dim=1)[0]
            pred_idx   = probs.argmax().item()
            pred_name  = LABEL_NAMES[pred_idx]
            confidence = probs[pred_idx].item()
            risk, emoji = RISK_LEVEL[pred_idx]

    # ── Results ───────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Result")
    st.markdown(f"**Prediction:** {pred_name}")
    st.markdown(f"**Confidence:** {confidence*100:.1f}%")
    st.markdown(f"**Risk level:** {emoji} {risk}")

    # ── Side by side: original + gradcam ─────────────────────
    st.subheader("Where the model looked")
    col1, col2 = st.columns(2)
    with col1:
        st.image(img.resize((224,224)), caption="Original", use_column_width=True)
    with col2:
        st.image(gradcam_viz, caption="Grad-CAM heatmap", use_column_width=True)

    st.caption("Red/warm regions = areas the model focused on to make its decision")

    # ── Confidence bar per class ──────────────────────────────
    st.subheader("Confidence across all classes")
    for idx, prob in enumerate(probs):
        st.write(LABEL_NAMES[idx])
        st.progress(float(prob))

    st.markdown("---")
    st.caption("⚠️ For educational purposes only. Not a substitute for medical diagnosis.")