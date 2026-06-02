import torch
from models_mae_cross_kan import mae_vit_base_patch16

device = "cuda" if torch.cuda.is_available() else "cpu"

model = mae_vit_base_patch16().to(device)
model.eval()

B = 2
imgs = torch.randn(B, 3, 384, 384).to(device)

# boxes: [B, shot_num, 3, 64, 64]
shot_num = 3
boxes = torch.randn(B, shot_num, 3, 64, 64).to(device)

with torch.no_grad():
    pred = model(imgs, boxes, shot_num)

print("Output shape:", pred.shape)