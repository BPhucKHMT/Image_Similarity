import streamlit as st
import cv2
import tempfile
import os
from glob import glob

# --- Import logic ---
from hist import extract_histogram, load_database, find_similar_images
from myFaiss import myFaiss
import torch
import clip
from PIL import Image

# --- Streamlit UI ---
st.title("🎨 Image Similarity Search Demo")

# Chọn phương pháp
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device="cpu")

method = st.radio("Chọn phương pháp tìm ảnh giống:", ("Histogram", "CLIP+Faiss"))

uploaded_file = st.file_uploader("Tải ảnh lên", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Lưu tạm file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    tfile.close()

    # Đọc ảnh
    image = cv2.imread(tfile.name)
    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Ảnh truy vấn", use_container_width=True)

    if method == "Histogram":
        # --- Histogram ---
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        query_hist = extract_histogram(image)
        dataset = load_database()
        results = find_similar_images(query_hist, dataset, top_k=10)

        st.subheader("🔍 Top 10 ảnh giống nhất (Histogram):")
        cols = st.columns(5)
        for i, (path, label, dist) in enumerate(results):
            path = path.replace("\\", "/") # Chuẩn hóa đường dẫn cho mọi OS
            with cols[i % 5]:
                img = cv2.imread(path)
                st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                         caption=f"{label} ({dist:.2f})", use_container_width=True)

    else:
        # --- CLIP + Faiss ---
  
        faiss_index = myFaiss(bin_clip_file="faiss_clip.bin", feature_shape=512, preprocess=preprocess, device=device)

        # Tải tất cả ảnh (dictionary {folder: [list ảnh]})
        data_dir = "dataset/seg"
        all_img_path = {}
        for folder in os.listdir(data_dir):
            folder_path = os.path.join(data_dir, folder)
            if os.path.isdir(folder_path):
                all_img_path[folder] = glob(os.path.join(folder_path, "*.jpg"))

        # Chuyển ảnh PIL để preprocess CLIP
        pil_img = Image.open(tfile.name).convert("RGB")

        # Search ảnh
        scores, idxs = faiss_index.image_search(pil_img, top_k=10)
        top_paths = faiss_index.Return_images(idxs, all_img_path)

        st.subheader("🔍 Top 10 ảnh giống nhất (CLIP+Faiss):")
        cols = st.columns(5)
        for i, (path, score) in enumerate(zip(top_paths, scores)):
            with cols[i % 5]:
                img = cv2.imread(path)
                st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                         caption=f"{os.path.basename(path)} ({score:.4f})", use_container_width=True)
