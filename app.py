# app.py
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import io
import numpy as np

# ---------------- App config ----------------
st.set_page_config(page_title="Lung Cancer Detection", page_icon="🫁", layout="centered")

# ---------------- Model architecture (must match training) ----------------
class DualResNet18(nn.Module):
    def __init__(self, num_types, num_stages=4):
        super(DualResNet18, self).__init__()
        base_model = models.resnet18(weights="IMAGENET1K_V1")
        # backbone excludes final fc
        self.backbone = nn.Sequential(*list(base_model.children())[:-1])  # feature extractor
        num_features = base_model.fc.in_features
        self.type_head = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_types)
        )
        self.stage_head = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_stages)
        )
    def forward(self, x):
        f = self.backbone(x)            # (B, C, 1, 1)
        f = f.view(f.size(0), -1)       # (B, C)
        return self.type_head(f), self.stage_head(f)

# ---------------- Canonical classes ----------------
CANONICAL_CLASSES = [
    "normal",
    "Bengin cases",
    "adenocarcinoma",
    "large.cell.carcinoma",
    "Malignant cases",
    "squamous.cell.carcinoma"
]

# ---------------- mapping from any model class name -> canonical class ----------------
def map_model_class_to_canonical(name: str) -> str:
    """
    Map a model's (possibly verbose) class folder name to one of the six canonical classes.
    This function is conservative and checks lowercase substrings.
    """
    n = name.lower()
    if "normal" in n:
        return "normal"
    if "adenocarcinoma" in n:
        return "adenocarcinoma"
    if "large.cell" in n or "large cell" in n:
        return "large.cell.carcinoma"
    if "squamous" in n:
        return "squamous.cell.carcinoma"
    # benign synonyms
    if "begin" in n or "benign" in n or "bengin" in n:
        return "Bengin cases"
    # malignant synonyms
    if "malign" in n:
        return "Malignant cases"
    # fallback: put in 'Bengin cases' if unknown? Better to put as 'Malignant cases' or 'normal'
    # we choose Unknown -> map to 'Bengin cases' conservatively to avoid false alarms.
    # If you'd prefer "Malignant cases" change below.
    return "Bengin cases"

# ---------------- image transform (match training) ----------------
def transform_image_pil(img_pil):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    return transform(img_pil).unsqueeze(0)

# ---------------- CT-likeness heuristic (fast, non-clinical) ----------------
def is_lung_ct_image(pil_img):
    """
    Heuristic check if an image is likely a chest CT slice.
    Returns (bool, reason_string). This reduces false classification on photos, docs, faces, etc.
    """
    try:
        img = pil_img.convert("L")
        arr = np.array(img).astype(np.float32)

        # low dynamic range -> not CT
        std = arr.std()
        if std < 8:
            return False, "Image has very low contrast (std < 8) — unlikely CT."

        # compute simple edge strength
        gx, gy = np.gradient(arr)
        edge_strength = np.mean(np.abs(gx) + np.abs(gy))
        if edge_strength < 4:
            return False, "Image has too few edges — unlikely CT."

        # corners vs center brightness: CT slices often show dark corners (scanner background)
        h, w = arr.shape
        corner_size_h = max(3, int(h * 0.12))
        corner_size_w = max(3, int(w * 0.12))
        corners = [
            arr[0:corner_size_h, 0:corner_size_w],
            arr[0:corner_size_h, -corner_size_w:],
            arr[-corner_size_h:, 0:corner_size_w],
            arr[-corner_size_h:, -corner_size_w:]
        ]
        corner_med = np.median([c.mean() for c in corners])
        center = arr[int(h*0.25):int(h*0.75), int(w*0.25):int(w*0.75)]
        center_mean = center.mean()

        # Accept if corners dark and center brighter, or edges strong and contrast adequate
        if corner_med < 40 and center_mean > 45:
            return True, "Corners dark & center brighter — likely CT slice."
        if edge_strength > 10 and std > 18:
            return True, "Strong edges and contrast — likely CT slice."
        return False, "Image does not match CT characteristics (contrast/edges/corners)."
    except Exception as e:
        return False, f"Validation error: {e}"

# ---------------- Load model (cached) ----------------
@st.cache_resource
def load_model_and_meta(checkpoint_path="lung_cancer_stage_model_best.pth", map_location="cpu"):
    """
    Loads checkpoint and returns:
       model (in eval mode), model_class_names (list)
    Accepts two checkpoint shapes:
      - dict with keys: model_state_dict, class_names (or class_to_idx), class_idx_to_stage
      - raw state_dict (then user must ensure correct class ordering below)
    """
    try:
        ckpt = torch.load(checkpoint_path, map_location=map_location)
    except Exception as e:
        st.error(f"Could not load checkpoint '{checkpoint_path}': {e}")
        return None

    # Determine saved state and metadata
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        saved_state = ckpt["model_state_dict"]
        # prefer explicit 'class_names' if present
        class_names = ckpt.get("class_names", None)
        class_to_idx = ckpt.get("class_to_idx", None)
        class_idx_to_stage = ckpt.get("class_idx_to_stage", None)
    else:
        # Assume ckpt is raw state dict, metadata missing
        saved_state = ckpt
        class_names = None
        class_to_idx = None
        class_idx_to_stage = None

    # If class_names missing, user must set correct ordering here.
    # We attempt to use class_to_idx if present; else fall back to a sensible default list the app expects.
    if class_names is None:
        if class_to_idx:
            # if class_to_idx is a dict mapping {class: idx}, produce list in idx order
            try:
                inv = sorted(class_to_idx.items(), key=lambda x: x[1])
                class_names = [c for c,_ in inv]
            except Exception:
                class_names = None
        if class_names is None:
            # DEFAULT FALLBACK: if your trained model used different class names,
            # please update this list to match the exact order used during training.
            # Here we provide a safe fallback (user should change if inaccurate).
            class_names = [
                'Bengin cases',
                'BenginCases',
                'Malignant cases',
                'MalignantCases',
                'adenocarcinoma',
                'adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib',
                'large.cell.carcinoma',
                'large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa',
                'normal',
                'squamous.cell.carcinoma',
                'squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa'
            ]

    num_types = len(class_names)
    num_stages = 4

    model = DualResNet18(num_types=num_types, num_stages=num_stages)
    # try strict load first, otherwise load non-strict
    try:
        model.load_state_dict(saved_state)
    except Exception:
        try:
            model.load_state_dict(saved_state, strict=False)
        except Exception as e:
            st.error(f"Failed to load model weights: {e}")
            return None

    model.eval()
    return {
        "model": model,
        "model_class_names": class_names,
        "class_idx_to_stage": class_idx_to_stage
    }

# ---------------- App UI ----------------
st.title("🫁 Lung Cancer Type & Stage Prediction")
st.write("Upload a chest CT slice (image). The app validates the image and shows prediction only if it's a likely CT slice.")

uploaded_file = st.file_uploader("Upload CT slice (png/jpg/jpeg/tif/tiff)", type=["png", "jpg", "jpeg", "tif", "tiff"])
st.sidebar.header("Settings")
ckpt_path = st.sidebar.text_input("Model checkpoint path", value="lung_cancer_stage_model_best.pth")

# load model (cached)
res = load_model_and_meta(ckpt_path)
if res is None:
    st.error("Model failed to load. Place your checkpoint file in the app folder and restart.")
    st.stop()

model = res["model"]
model_class_names = res["model_class_names"]
class_idx_to_stage = res.get("class_idx_to_stage", None)

# transform and stage label map
stage_label_map = {0: "Stage 0 (Normal)", 1: "Stage I", 2: "Stage II", 3: "Stage III/IV"}

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
    except Exception as e:
        st.error(f"Could not read uploaded file: {e}")
        st.stop()

    st.image(image, caption="Uploaded image", use_column_width=True)

    # validate if image looks like CT slice
    valid, reason = is_lung_ct_image(image)
    if not valid:
        st.error("This image does not appear to be a valid chest CT slice.\nReason: " + reason)
    else:
        st.success("Image validation passed: " + reason)

        if st.button("Classify Image"):
            x = transform_image_pil(image)  # (1,C,H,W)
            with torch.no_grad():
                type_logits, stage_logits = model(x)
                type_probs = F.softmax(type_logits, dim=1).cpu().numpy()[0]
                stage_probs = F.softmax(stage_logits, dim=1).cpu().numpy()[0]

            # aggregate model class probabilities into the 6 canonical classes
            canonical_probs = {c: 0.0 for c in CANONICAL_CLASSES}
            # iterate model class names and add their probability to the mapped canonical bucket
            for idx, orig_name in enumerate(model_class_names):
                p = float(type_probs[idx])
                mapped = map_model_class_to_canonical(orig_name)
                if mapped not in canonical_probs:
                    # safety: if mapping returned unknown, put into 'Bengin cases'
                    mapped = "Bengin cases"
                canonical_probs[mapped] += p

            # normalize (should already sum ~1.0) and convert to percentages
            total = sum(canonical_probs.values())
            if total <= 0:
                total = 1.0
            for k in canonical_probs:
                canonical_probs[k] = canonical_probs[k] / total

            # predicted canonical class and model confidence
            pred_canonical = max(canonical_probs.items(), key=lambda x: x[1])[0]
            pred_conf = canonical_probs[pred_canonical] * 100.0

            # stage predicted by stage head
            pred_stage_idx = int(np.argmax(stage_probs))
            pred_stage_label = stage_label_map.get(pred_stage_idx, f"Stage {pred_stage_idx}")

            # optional: if checkpoint had class_idx_to_stage mapping, show it
            mapped_stage_text = None
            if class_idx_to_stage is not None:
                # try to fetch mapped stage for the predicted original class index (best single orig)
                try:
                    best_orig_idx = int(np.argmax(type_probs))
                    # class_idx_to_stage keys may be ints or strings; handle robustly
                    mapped_val = class_idx_to_stage.get(best_orig_idx, class_idx_to_stage.get(str(best_orig_idx)))
                    if mapped_val is not None:
                        mapped_stage_text = f"Mapped from class folder → Stage {mapped_val}"
                except Exception:
                    mapped_stage_text = None

            # ---------------- DISPLAY RESULTS (ONLY CANONICAL) ----------------
            st.markdown("## Prediction Result")
            st.write(f"**Predicted Cancer Type:** `{pred_canonical}`")
            st.write(f"**Predicted Stage (model):** `{pred_stage_label}`")
            if mapped_stage_text:
                st.write(f"**Stage mapping (from class folder):** {mapped_stage_text}")
            st.write(f"**Model Confidence:** `{pred_conf:.2f}%`")

            # show canonical probabilities only (collapsed list)
            st.markdown("### Canonical class probabilities (collapsed):")
            for cname in CANONICAL_CLASSES:
                st.write(f"- {cname}: {canonical_probs.get(cname, 0.0) * 100.0:.2f}%")

            st.info("Note: This tool is for research/demo only and not for clinical use.")
