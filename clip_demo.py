import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# 画像とテキストの準備
image = preprocess(Image.open("Data/1_wan.jpg")).unsqueeze(0).to(device)
list = ["1_wan", "2_wan", "3_wan", "4_wan", "5_wan", "6_wan", "7_wan", "8_wan", "9_wan", "1_man"]
text = clip.tokenize(list).to(device)

with torch.no_grad():
    # 画像とテキストのエンコード
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)

    # 推論
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

# 類似率の出力
print("Label probs:", probs)