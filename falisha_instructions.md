# Falisha GL23x45 dTEC Dataset Handoff

## What This Dataset Is

This is the final windowed training dataset for Falisha's SFNO model.

The model-facing tensors are:

```python
tec_input   [batch, 6, 23, 45]
omni_input  [batch, 6, 5]
target      [batch, 3, 23, 45]
```

The TEC field is residual dTEC:

```text
dTEC = IONEX vTEC on GL23x45 - IRI vTEC on GL23x45
```

The GL grid uses:

```text
Lmax = 22
nlat = 23
nlon = 45
```

The plasmaspheric offset is intentionally retained for the baseline dataset. It is treated as a known mostly zonal systematic that the SFNO can represent through its m=0 modes.

## Dataset Location

The final dataset lives here:

```text
data/falisha_windows_gl23x45/
```

That folder contains normalized split arrays:

```text
train_tec_input.npy
train_omni_input.npy
train_target.npy
train_window_start_times.npy

val_tec_input.npy
val_omni_input.npy
val_target.npy
val_window_start_times.npy

test_tec_input.npy
test_omni_input.npy
test_target.npy
test_window_start_times.npy

lats.npy
lons.npy
metadata.json
```

## Split Sizes

The validated split sizes are:

```text
train: 110,124 windows
val:    25,406 windows
test:   26,270 windows
```

Each split has these shapes:

```python
train_tec_input.shape   == (110124, 6, 23, 45)
train_omni_input.shape  == (110124, 6, 5)
train_target.shape      == (110124, 3, 23, 45)

val_tec_input.shape     == (25406, 6, 23, 45)
val_omni_input.shape    == (25406, 6, 5)
val_target.shape        == (25406, 3, 23, 45)

test_tec_input.shape    == (26270, 6, 23, 45)
test_omni_input.shape   == (26270, 6, 5)
test_target.shape       == (26270, 3, 23, 45)
```

## Normalization

The arrays are already normalized.

Normalization statistics were computed from train inputs only:

```text
tec_input train split -> TEC mean/std
omni_input train split -> OMNI mean/std
```

The TEC normalization is applied to both:

```text
tec_input
target
```

The OMNI normalization is applied to:

```text
omni_input
```

The exact normalization values are stored in:

```text
data/falisha_windows_gl23x45/metadata.json
```

## OMNI Feature Order

The OMNI feature order is:

```python
[
    "b_magnitude",
    "by_gsm",
    "bz_gsm",
    "flow_speed",
    "proton_density",
]
```

So:

```python
omni_input[:, :, 0]  # b_magnitude
omni_input[:, :, 1]  # by_gsm
omni_input[:, :, 2]  # bz_gsm
omni_input[:, :, 3]  # flow_speed
omni_input[:, :, 4]  # proton_density
```

## PyTorch Access

Use this loader:

```python
from data_pull.falisha_dataset import make_falisha_dataloader

train_loader = make_falisha_dataloader(
    root="data/falisha_windows_gl23x45",
    split="train",
    batch_size=8,
    num_workers=0,
)

batch = next(iter(train_loader))

tec_input = batch["tec_input"]
omni_input = batch["omni_input"]
target = batch["target"]
timestamp = batch["timestamp"]

print(tec_input.shape)   # torch.Size([8, 6, 23, 45])
print(omni_input.shape)  # torch.Size([8, 6, 5])
print(target.shape)      # torch.Size([8, 3, 23, 45])
```

## Direct Dataset Access

If you do not want a DataLoader:

```python
from data_pull.falisha_dataset import FalishaDTECDataset

ds = FalishaDTECDataset(
    root="data/falisha_windows_gl23x45",
    split="train",
)

sample = ds[0]

print(sample["tec_input"].shape)    # torch.Size([6, 23, 45])
print(sample["omni_input"].shape)   # torch.Size([6, 5])
print(sample["target"].shape)       # torch.Size([3, 23, 45])
print(sample["timestamp"])          # window start time, epoch seconds
```

## Available Splits

Use one of:

```python
split="train"
split="val"
split="test"
```

By default, `make_falisha_dataloader()` shuffles only the train split.

## Window Definition