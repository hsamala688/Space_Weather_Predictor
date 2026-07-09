from data_pull.falisha_dataset import make_falisha_dataloader

loader = make_falisha_dataloader(
    root="data/falisha_windows_gl23x45",
    split="train",
    batch_size=8,
    num_workers=0,
)

batch = next(iter(loader))
print(batch["tec_input"].shape)   # [8, 6, 23, 45]
print(batch["omni_input"].shape)  # [8, 6, 5]
print(batch["target"].shape)      # [8, 3, 23, 45]