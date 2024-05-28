from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import faiss
import numpy as np
import os
from typing import List
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
class AddImage(BaseModel):
    image: str
    image_name: str
class SearchImage(BaseModel):
    image: str
class Cleardata(BaseModel):
    clear: str

images = []
app = FastAPI()
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
processor = AutoImageProcessor.from_pretrained('facebook/dinov2-small')
model = AutoModel.from_pretrained('facebook/dinov2-small').to(device)

@app.get("/")
async def read_root():
    return {"Hello": "World"}

@app.post("/add_image")
async def add_image(image: List[AddImage]):
    for i in image:
        image = base64.b64decode(i.image)
        with open(f"images/{i.image_name}.jpg", "wb") as f:
            f.write(image)
    
    return {"status": "success"}



@app.post("/clear_data")
async def clear_data(data: Cleardata):

    for root, dirs, files in os.walk('./images'):
        for file in files:
            os.remove(root  + '/'+ file)
    return {"status": "cleared"}

@app.post("/search_image")
async def search_image(data: SearchImage):
# compile all images in the folder
    for root, dirs, files in os.walk('./images'):
        for file in files:
            if file.endswith('jpg'):
                images.append(root  + '/'+ file)
            if file.endswith('png'):
                images.append(root  + '/'+ file)

    def add_vector_to_index(embedding, index):
        vector = embedding.detach().cpu().numpy()
        vector = np.float32(vector)
        faiss.normalize_L2(vector)
        index.add(vector)

    index = faiss.IndexFlatL2(384)

    for image_path in images:
        img = Image.open(image_path).convert('RGB')
        with torch.no_grad():
            inputs = processor(images=img, return_tensors="pt").to(device)
            outputs = model(**inputs)
        features = outputs.last_hidden_state
        add_vector_to_index( features.mean(dim=1), index)

    faiss.write_index(index,"vector.index")

# search for the image

    image = base64.b64decode(data.image)
    with open("search.jpg", "wb") as f:
        f.write(image)
    img = Image.open("search.jpg").convert('RGB')
    with torch.no_grad():
        inputs = processor(images=img, return_tensors="pt").to(device)
        outputs = model(**inputs)
    
    embeddings = outputs.last_hidden_state
    embeddings = embeddings.mean(dim=1)
    vector = embeddings.detach().cpu().numpy()
    vector = np.float32(vector)
    faiss.normalize_L2(vector)

    index = faiss.read_index("vector.index")
    print(images)
    print(len(images))
    if len(images)<5:
        k= len(images)
    else:
        k=5
    d,i = index.search(vector,k)
    print('Images:', [images[index] for index in i[0]])

    # Convert images to base64
    base64_images = []
    for index in i[0]:
        with open(images[index], "rb") as img_file:
            base64_images.append(base64.b64encode(img_file.read()).decode('utf-8'))

    return {"status": "success", "result": base64_images}
