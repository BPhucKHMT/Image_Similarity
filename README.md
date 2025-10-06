# Image_Similarity
## How to start
pip install -r requirements.txt
open terminal
streamlit run app.py

## Structure
'''
project_root/
│
├── 📂 dataset/ # dataset of images grouped by class
│ ├── buildings/
│ ├── forest/
│ └── ...
│
├── 📄 image_hist.db # SQLite database storing image histograms
│
├── 📄 myFaiss.py # build & search FAISS index
│
├── 📄 hist.py # extract color histogram features and save to DB
│
├── 📒 Histogram_Lab02.ipynb # CS406 Lab02: extract histogram feature +
│
├── 📒 Clip.ipynb # CLIP-based retrieval or FAISS demo notebook
│
└── 📄 requirements.txt # project dependencies
'''
