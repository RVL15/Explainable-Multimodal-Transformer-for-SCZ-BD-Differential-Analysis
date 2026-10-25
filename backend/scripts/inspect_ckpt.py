import torch

ckpt = torch.load("backend/checkpoints/best_model.pt", map_location="cpu")
print("Target task:", ckpt.get("target_task"))
print("Epoch:", ckpt.get("epoch"))
print("Val acc:", ckpt.get("val_acc"))
print("Class mapping:", ckpt.get("class_mapping"))
print("Hyperparameters:", ckpt.get("hyperparameters"))
splits = ckpt.get("split_participant_ids", {})
print("Splits summary:")
for k, v in splits.items():
    print(f"  {k}: count={len(v)}, sample={v[:3]}")

norm_mean = torch.tensor(ckpt.get("norm_mean"))
norm_std = torch.tensor(ckpt.get("norm_std"))
print("Norm mean shape:", norm_mean.shape, "min/max:", norm_mean.min().item(), norm_mean.max().item())
print("Norm std shape:", norm_std.shape, "min/max:", norm_std.min().item(), norm_std.max().item())
print("Preprocessing config:", ckpt.get("preprocessing_config"))
