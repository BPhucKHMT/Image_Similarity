
# Import module
import os
import clip
import glob
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import faiss
import json


class myFaiss:
    def __init__(self, bin_clip_file = "faiss_clip.bin", feature_shape=512, preprocess=None, device =None):
        self.index_clip = self.load_bin_file(bin_clip_file)
        self.clip_model = clip.load("ViT-B/32", device=device)[0]
        self.__device = "cuda" if torch.cuda.is_available() else "cpu"
        self.preprocess = preprocess
        self.clip_model.to(self.__device)

    def load_bin_file(self, bin_file):
        return faiss.read_index(bin_file)
    def image_search(self, img, top_k=5):
        img = self.preprocess(img).unsqueeze(0).to(self.__device)
        with torch.no_grad():
            image_features = self.clip_model.encode_image(img)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        image_features = image_features.cpu().numpy().astype(np.float32)
        scores , idx_image = self.index_clip.search(image_features, top_k)
        idx_image = idx_image.flatten()
        scores = scores.flatten()
        return scores , idx_image
    def Return_images (self, idxs, all_img_path):
        paths = []
        id2img = json.load(open('id2img.json', 'r'))
        for i in idxs:
            idx = str(i)
            img_path = id2img[idx]
            paths.append(img_path)
        return paths
