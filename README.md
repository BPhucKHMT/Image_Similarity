# Image_Similarity
## How to start
- pip install -r requirements.txt
- open terminal
- streamlit run app.py

## Structure
```
project_root/
│
├── 📂 dataset/ # dataset of images grouped by class
│ ├── buildings/
│ ├── forest/
│ └── ...
│
├── 📄 image_hist.db # SQLite database storing image histograms
│
├── 📄 myFaiss.py #  search FAISS index
│
├── 📄 hist.py # extract color histogram features and save to DB
│
├── 📒 23521208_Histogram_Lab02.ipynb # CS406 Lab02: extract histogram feature + find similar images
│
├── 📒 Clip.ipynb #  build FAISS demo notebook ( include create id2img.json and bin file)
│
└── 📄 requirements.txt # project dependencies
```



