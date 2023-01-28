import io
import requests
from PIL import Image
import torch

import japanese_clip as ja_clip
device = "cuda" if torch.cuda.is_available() else "cpu"
# ja_clip.available_models()
# ['rinna/japanese-clip-vit-b-16', 'rinna/japanese-cloob-vit-b-16']
# If you want v0.1.0 models, set `revision='v0.1.0'`
model, preprocess = ja_clip.load("rinna/japanese-clip-vit-b-16", cache_dir="/tmp/japanese_clip", device=device)
tokenizer = ja_clip.load_tokenizer()

list = ["一", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
image = preprocess(Image.open("Data/1_wan.jpg")).unsqueeze(0).to(device)
encodings = ja_clip.tokenize(
    texts=list,
    max_seq_len=77,
    device=device,
    tokenizer=tokenizer, # this is optional. if you don't pass, load tokenizer each time
)

with torch.no_grad():
    image_features = model.get_image_features(image)
    text_features = model.get_text_features(**encodings)
    text_probs = (100 * image_features @ text_features.T).softmax(dim=-1)

print("Label probs:", text_probs)