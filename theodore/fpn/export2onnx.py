# %%
import torch
import segmentation_models_pytorch as smp

MODEL_PATH = './runs/pan_mobilenet_v2_schedule_25_upd_early_stopping/best.pth'

# %%
best_model = torch.load(MODEL_PATH)
print(sum(p.numel() for p in best_model.parameters() if p.requires_grad))


# %%
best_model.eval()
torch_input = torch.randn(1, 3, 128, 128).to("cuda").float()
torch.onnx.export(
    best_model,
    torch_input,
    MODEL_PATH.replace(".pth", "_upd.onnx"),
    input_names=["input"],
    output_names=["output"],
    verbose=False,
)

# %%



