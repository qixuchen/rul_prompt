import torch
import clip
from PIL import Image
import pickle

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

prompt_t1 = 'the remaining useful life is'
prompt_t1s = [prompt_t1 + ' {:}'.format(x) for x in range(126)]

text = clip.tokenize(prompt_t1s).to(device)

with torch.no_grad():
    text_features = model.encode_text(text)

res_features = text_features.cpu().numpy()
f = open('./feats/clip_feature_ts_forcasting.pkl', 'wb')
# res = {'data': res_features}
pickle.dump(res_features,f)

