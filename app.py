import streamlit as st
import cv2
import numpy as np
import os
from glob import glob
import torch
import clip
from PIL import Image
from io import BytesIO

# --- Import logic ---
from hist import extract_histogram, load_database, find_similar_images
from myFaiss import myFaiss

# --- Streamlit UI ---
st.title("🎨 Image Similarity Search Demo")

# Chọn thiết bị và model CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)

method = st.radio("Chọn phương pháp tìm ảnh giống:", ("Histogram", "CLIP+Faiss"))

uploaded_file = st.file_uploader("Tải ảnh lên", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Đọc ảnh trực tiếp từ memory (không ghi ra file)
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        st.error("Không đọc được ảnh upload.")
        st.stop()

    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
             caption="Ảnh truy vấn", width='stretch')

    if method == "Histogram":
        # --- Histogram ---
        rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        query_hist = extract_histogram(rgb_img)
        dataset = load_database()
        results = find_similar_images(query_hist, dataset, top_k=10)

        st.subheader("🔍 Top 10 ảnh giống nhất (Histogram):")
        cols = st.columns(5)

        for i, (path, label, dist) in enumerate(results):
            path = path.replace("\\", "/")
            if not os.path.exists(path):
                continue
            img = cv2.imread(path)
            if img is None:
                continue
            with cols[i % 5]:
                st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                         caption=f"{label} ({dist:.2f})", width='stretch')

    else:
        # --- CLIP + Faiss ---
        data_dir = "dataset/seg"
        if not os.path.exists("faiss_clip.bin"):
            st.error("❌ Thiếu file faiss_clip.bin — cần build index trước.")
            st.stop()

        faiss_index = myFaiss(
            bin_clip_file="faiss_clip.bin",
            feature_shape=512,
            preprocess=preprocess,
            device=device
        )

        # Tải dataset
        all_img_path = {}
        for folder in os.listdir(data_dir):
            folder_path = os.path.join(data_dir, folder)
            if os.path.isdir(folder_path):
                all_img_path[folder] = glob(os.path.join(folder_path, "*.jpg"))

        # Chuyển ảnh sang PIL để CLIP preprocess
        pil_img = Image.open(BytesIO(file_bytes)).convert("RGB")

        # Search ảnh
        scores, idxs = faiss_index.image_search(pil_img, top_k=10)
        top_paths = faiss_index.Return_images(idxs, all_img_path)

        st.subheader("🔍 Top 10 ảnh giống nhất (CLIP+Faiss):")
        cols = st.columns(5)

        for i, (path, score) in enumerate(zip(top_paths, scores)):
            path = path.replace("\\", "/")
            if not os.path.exists(path):
                continue
            img = cv2.imread(path)
            if img is None:
                continue
            with cols[i % 5]:
                st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                         caption=f"{os.path.basename(path)} ({score:.4f})",
                         width='stretch')
