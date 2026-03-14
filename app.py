
# ============================================================
# MINERAL CLASSIFICATION SYSTEM
# DINOv2 + MLP + MAHALANOBIS OOD
# STREAMLIT APPLICATION
# ============================================================
import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np
from PIL import Image
import pickle
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AUTHORS_DIR = os.path.join(BASE_DIR, "authors")
# ------------------------------------------------------------
# ------------------------------------------------------------
# DEVICE
# ------------------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------------------------------------
# TORCH HUB CACHE (important for deployment)
# ------------------------------------------------------------

torch.hub.set_dir("torch_cache")

# ------------------------------------------------------------
# CLASSES
# ------------------------------------------------------------

CLASSES = ["stone","gem_mineral_ore","mineral","rock"]

IMG_SIZE = 224

# ------------------------------------------------------------
# IMAGE TRANSFORM
# ------------------------------------------------------------

transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(
        mean=(0.485,0.456,0.406),
        std=(0.229,0.224,0.225)
    )
])

# ------------------------------------------------------------
# LOAD DINOv2
# ------------------------------------------------------------

@st.cache_resource
def load_dino():

    model = torch.hub.load(
        "facebookresearch/dinov2",
        "dinov2_vits14",
        trust_repo=True
    )

    model.eval()
    model.to(DEVICE)

    return model

dino = load_dino()
# ------------------------------------------------------------
# MLP CLASSIFIER
# ------------------------------------------------------------

class MLP_MMD(nn.Module):

    def __init__(self):

        super().__init__()

        self.fc1 = nn.Linear(384,256)
        self.bn1 = nn.BatchNorm1d(256)

        self.fc2 = nn.Linear(256,64)
        self.bn2 = nn.BatchNorm1d(64)

        self.classifier = nn.Linear(64,4)

        self.act = nn.GELU()
        self.drop = nn.Dropout(0.2)

    def forward(self,x):

        x = self.drop(self.act(self.bn1(self.fc1(x))))
        feat = self.act(self.bn2(self.fc2(x)))

        logits = self.classifier(feat)

        return logits

# ------------------------------------------------------------
# LOAD TRAINED CLASSIFIER
# ------------------------------------------------------------

@st.cache_resource
def load_classifier():

    if not os.path.exists("best_model.pth"):
        st.error("best_model.pth not found in project folder")
        st.stop()

    model = MLP_MMD().to(DEVICE)

    model.load_state_dict(
        torch.load("best_model.pth", map_location=DEVICE)
    )

    model.eval()

    return model

model = load_classifier()

# ------------------------------------------------------------
# LOAD OOD STATISTICS
# ------------------------------------------------------------

@st.cache_resource
def load_stats():

    if not os.path.exists("ood_stats.pkl"):
        st.error("ood_stats.pkl not found in project folder")
        st.stop()

    with open("ood_stats.pkl","rb") as f:
        stats = pickle.load(f)

    means = stats["means"]
    inv_cov = stats["inv_cov"]
    threshold = stats["threshold"]

    return means, inv_cov, threshold


means, inv_cov, OOD_THRESHOLD = load_stats()

# ------------------------------------------------------------
# FEATURE EXTRACTION
# ------------------------------------------------------------

def extract_feature(image):

    img = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        feat = dino(img)

    feat = torch.nn.functional.normalize(feat,p=2,dim=1)

    return feat.cpu().numpy()[0]

# ------------------------------------------------------------
# MAHALANOBIS DISTANCE
# ------------------------------------------------------------

def mahalanobis(x, mean):

    diff = x - mean

    return np.sqrt(diff.T @ inv_cov @ diff)

# ------------------------------------------------------------
# CLASSIFICATION
# ------------------------------------------------------------

def classify_feature(feature):

    x = torch.tensor(feature).unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        logits = model(x)
        probs = torch.softmax(logits,dim=1)
        pred = torch.argmax(probs,dim=1).item()

    label = CLASSES[pred]

    d = mahalanobis(feature, means[label])

    if d > OOD_THRESHOLD:
        label = "unknown"

    return label, probs.cpu().numpy()[0]


# ============================================================
# STREAMLIT USER INTERFACE
# ============================================================

# ------------------------------------------------------------
# PATH CONFIGURATION
# ------------------------------------------------------------

SAMPLES_DIR = os.path.join(BASE_DIR, "samples")

# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------

st.set_page_config(
    page_title="Mineralogical Material Detection",
    layout="wide"
)

# ------------------------------------------------------------
# GLOBAL FONT SIZE FIX (READABILITY)
# ------------------------------------------------------------
st.markdown("""
<style>

html, body, [class*="css"] {
    font-size:26px;
}

h1 {
    font-size:80px !important;
    font-weight:700;
}

h2 {
    font-size:52px !important;
}

h3 {
    font-size:40px !important;
}

p, li {
    font-size:26px !important;
    line-height:1.7;
}

.navbar{
text-align:center;
font-size:28px;
margin-top:16px;
margin-bottom:16px;
}

.navbar a{
text-decoration:none;
color:#333;
padding:12px 24px;
font-weight:700;
}

.navbar a:hover{
color:#0d6efd;
border-bottom:4px solid #0d6efd;
}

</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# HEADER (INSTITUTE LOGO + TITLE + TEXMiN LOGO)
# ------------------------------------------------------------
col1, col2, col3 = st.columns([1,6,1])

# Institute Logo (Left)
with col1:
    inst_logo = os.path.join(BASE_DIR, "nitn.logo.png")
    if os.path.exists(inst_logo):
        st.image(inst_logo, width=200)

# Title (Center)
with col2:
    st.markdown(
        "<h1 style='text-align:center;'>Mineralogical Material Detection</h1>",
        unsafe_allow_html=True
    )

# TEXMiN Logo (Right)
with col3:
    texmin_logo = os.path.join(BASE_DIR, "texmin_logo.png")
    if os.path.exists(texmin_logo):
        st.image(texmin_logo, width=250)

# ------------------------------------------------------------
# NAVIGATION BAR
# ------------------------------------------------------------

st.markdown("""

<div class="navbar">

<a href="?section=home">Home</a>
|
<a href="?section=contact">Contact</a>
|
<a href="?section=author">Author</a>

</div>

""", unsafe_allow_html=True)

st.markdown("<hr style='border:2px dotted gray'>", unsafe_allow_html=True)

# ------------------------------------------------------------
# READ PAGE SECTION
# ------------------------------------------------------------

query_params = st.query_params
section = query_params.get("section", "home")

# ============================================================
# HOME SECTION
# ============================================================

if section == "home":

    st.subheader("Mineral Image Input")

    image = None

    input_mode = st.radio(
        "Choose Input Method",
        ["Dropdown", "Camera", "Upload"],
        horizontal=True
    )

    # --------------------------------
    # DROPDOWN
    # --------------------------------

    if input_mode == "Dropdown":

        samples = {

        "mineral (epidote)": ("mineral","epidote.png"),
        "stone (perovskite)": ("stone","perovskite.png"),
        "rock (chondrite)": ("rock","chondrite.png"),
        "gem_mineral_ore (gypsum)": ("gem_mineral_ore","gypsum.png"),
        "mineral (Biotite)": ("mineral","biotite.png"),
        "rock (allitroclastic breccia)": ("rock","allogeneic breccia.png"),
        "stone (meteorite seimchan)": ("stone","meteorite seimchan.png")

        }

        option = st.selectbox(
            "Select mineral sample",
            ["None"] + list(samples.keys())
        )

        if option != "None":

            class_label, file_name = samples[option]

            sample_path = os.path.join(SAMPLES_DIR, file_name)

            if os.path.exists(sample_path):

                image = Image.open(sample_path).convert("RGB")
                st.image(image, width=350)

            else:
                st.error(f"Image not found: {sample_path}")

    # --------------------------------
    # CAMERA
    # --------------------------------

    elif input_mode == "Camera":

        cam = st.camera_input("Capture mineral image")

        if cam is not None:
            image = Image.open(cam).convert("RGB")

    # --------------------------------
    # UPLOAD
    # --------------------------------

    elif input_mode == "Upload":

        uploaded = st.file_uploader(
            "Upload mineral image",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded:
            image = Image.open(uploaded).convert("RGB")

    st.markdown("<hr style='border:2px dotted gray'>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # PROCESSING + RESULT
    # --------------------------------------------------------

    if image is not None:

        st.image(image, width=400)

        with st.spinner("Processing mineral image..."):

            feature = extract_feature(image)
            label, probs = classify_feature(feature)

        st.success("Processing complete")

        st.subheader("Result")

        col_img, col_res = st.columns([1,2])

        with col_img:
            st.image(image, width=300)

        with col_res:

            st.markdown(f"### Prediction: **{label}**")

            if label != "unknown":

                st.markdown("#### Class Probabilities")

                for c, p in zip(CLASSES, probs):

                    st.write(f"**{c}**")
                    st.progress(float(p))
                    st.write(f"{p*100:.2f}%")

            else:
                st.error("Out-of-distribution mineral sample detected")

        # --------------------------------------------------------
        # ABOUT PROJECT
        # --------------------------------------------------------

        st.markdown("<hr style='border:2px dotted gray'>", unsafe_allow_html=True)

        st.subheader("About the Project")

        st.write("""
This application implements an automated **Mineralogical Material Detection**.

The system utilizes a **DINOv2 Vision Transformer encoder** to extract semantic
features from mineral images. These features are processed by a 
**MMD-MLP classifier** trained to categorize geological samples into four classes:

• stone  
• mineral  
• rock  
• gem_mineral_ore  

To ensure reliability, the system incorporates **Mahalanobis distance-based
Out-of-Distribution (OOD) detection**, which identifies mineral images that do
not belong to the trained distribution and labels them as **unknown**.

Images can be provided through three input modes:

• Dropdown sample selection  
• Camera capture  
• Image upload  

After input, the system performs **feature extraction → classification → OOD
verification** before presenting the predicted mineral class and probability
distribution.
""")

        # --------------------------------------------------------
        # ACKNOWLEDGEMENT
        # --------------------------------------------------------

        st.markdown("<hr style='border:2px dotted gray'>", unsafe_allow_html=True)

        st.subheader("Acknowledgement")

        st.write("""
This project was supported under the **TEXMiN UG/PG Fellowship**, a Technology
Innovation Hub at **IIT (ISM) Dhanbad**, and carried out under an institutional
**MoU collaboration with the National Institute of Technology Nagaland**.
""")

        st.markdown(
            "<h3 style='text-align:center;color:green;'>Thank You</h3>",
            unsafe_allow_html=True
        )

# ============================================================
# AUTHOR SECTION
# ============================================================

elif section == "author":

    st.subheader("Author")

    a1, a2, a3 = st.columns(3)

    with a1:
        st.image(os.path.join(AUTHORS_DIR,"Chomangli1.jpg"), width=350)
        st.markdown("**Chomangli T Sangtam**")
        st.write("B.Tech Student")
        st.write("Department of Electrical and Electronics Engineering")
        st.write("National Institute of Technology Nagaland")

    with a2:
        st.image(os.path.join(AUTHORS_DIR,"maloy1.jpg"), width=350)
        st.markdown("**Mr. Maloy Kumar Dey**")
        st.write("PhD Scholar")
        st.write("Department of Computer Science and Engineering")
        st.write("National Institute of Technology Nagaland")

    with a3:
        st.image(os.path.join(AUTHORS_DIR,"dushman1.jpg"), width=350)
        st.markdown("**Dr. Dushman Kumar Das**")
        st.write("Associate Professor")
        st.write("Department of Electrical and Electronics Engineering")
        st.write("National Institute of Technology Nagaland")

# ============================================================
# CONTACT SECTION
# ============================================================

elif section == "contact":

    st.subheader("Contact")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**Chomangli T Sangtam**")
        st.write("Phone: 7005182117")
        st.write("Email: chomanglisangtam761@gmail.com")

    with c2:
        st.markdown("**Mr. Maloy Kumar Dey**")
        st.write("Phone: 8296810118")
        st.write("Email: maloy24@gmail.com")

    with c3:
        st.markdown("**Dr. Dushman Kumar Das**")
        st.write("Phone: 9861060880")
        st.write("Email: dushmantakumardas29@gmail.com")