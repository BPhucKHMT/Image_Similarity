import sqlite3
import numpy as np
import cv2

# --- Hàm tính histogram ---
def extract_histogram(image):
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8],
                        [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

# --- Hàm lấy dữ liệu từ DB ---
def load_database(db_path="image_hist.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT path, label, histogram FROM images")
    data = cursor.fetchall()
    conn.close()
    dataset = []
    for path, label, hist_blob in data:
        hist = np.frombuffer(hist_blob, dtype=np.float32)
        dataset.append((path, label, hist))
    return dataset

# --- Tính khoảng cách Euclid ---
def find_similar_images(query_hist, dataset, top_k=10):
    results = []
    for path, label, hist in dataset:
        dist = np.linalg.norm(query_hist - hist)
        results.append((path, label, dist))
    results.sort(key=lambda x: x[2])
    return results[:top_k]
