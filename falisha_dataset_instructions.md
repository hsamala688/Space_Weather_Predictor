# Falisha GL23x45 dTEC Dataset Handoff

## Overview

This is the final windowed dataset for Falisha's SFNO training pipeline.

The model-facing tensors are:

```python
tec_input   [batch, 6, 23, 45]
omni_input  [batch, 6, 6]
target      [batch, 3, 23, 45]
```

The TEC field is residual dTEC:

```text
dTEC = IONEX vTEC on GL23x45 - IRI vTEC on GL23x45
```

The grid is:

```text
Lmax = 22
nlat = 23
nlon = 45
```

The plasmaspheric offset is intentionally retained in the baseline residual. It is treated as a known mostly zonal systematic that the SFNO can represent through its m=0 modes.

## Delivery

The dataset is shared separately through Google Drive as:

```text
falisha_windows_gl23x45.tar.gz
```

The Git repository contains the code and loader. The dataset arrays are not stored in normal Git history.

Download the archive from Google Drive, place it at the repository root, then unpack it:

```bash
cd Space_Weather_Predictor
tar -xzf falisha_windows_gl23x45.tar.gz
```

After unpacking, this folder should exist:

```text
data/falisha_windows_gl23x45/
```

The PyTorch loader assumes that path by default.

## Dataset Contents

The dataset folder contains:

```text
data/falisha_windows_gl23x45/
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

The arrays are standard NumPy `.npy` files and are intended to be read with memory mapping.

## Split Sizes

The validated split sizes are:

```text
train: 110,124 windows
val:    25,406 windows
test:   26,270 windows
```

Expected array shapes:

```python
train_tec_input.shape   == (110124, 6, 23, 45)
train_omni_input.shape  == (110124, 6, 6)
train_target.shape      == (110124, 3, 23, 45)

val_tec_input.shape     == (25406, 6, 23, 45)
val_omni_input.shape    == (25406, 6, 6)
val_target.shape        == (25406, 3, 23, 45)

test_tec_input.shape    == (26270, 6, 23, 45)
test_omni_input.shape   == (26270, 6, 6)
test_target.shape       == (26270, 3, 23, 45)
```

## Driver Feature Order

`omni_input` contains six driver channels:

```python
[
    "b_magnitude",
    "by_gsm",
    "bz_gsm",
    "flow_speed",
    "proton_density",
    "kp_3hour",
]
```

Index mapping:

```python
omni_input[:, :, 0]  # b_magnitude
omni_input[:, :, 1]  # by_gsm
omni_input[:, :, 2]  # bz_gsm
omni_input[:, :, 3]  # flow_speed
omni_input[:, :, 4]  # proton_density
omni_input[:, :, 5]  # kp_3hour
```

`kp_3hour` is causal 3-hourly Kp. For each dTEC timestamp, it is the most recent 3-hour Kp value at or before that timestamp. It is forward-filled, not linearly interpolated, so it does not use future Kp values.

## Normalization

The arrays are already normalized.

Normalization statistics were computed from the training inputs only:

```text
train tec_input   -> TEC mean/std
train omni_input  -> driver mean/std
```

TEC normalization is applied to:

```text
tec_input
target
```

Driver normalization is applied to:

```text
omni_input
```

Do not normalize again unless intentionally changing preprocessing.

The exact normalization values are stored in:

```text
data/falisha_windows_gl23x45/metadata.json
```

## PyTorch Loader

Use:

```python
from data_pull.falisha_dataset import make_falisha_dataloader
```

Create loaders:

```python
train_loader = make_falisha_dataloader(
    root="data/falisha_windows_gl23x45",
    split="train",
    batch_size=8,
    num_workers=0,
)

val_loader = make_falisha_dataloader(
    root="data/falisha_windows_gl23x45",
    split="val",
    batch_size=8,
    num_workers=0,
    shuffle=False,
)

test_loader = make_falisha_dataloader(
    root="data/falisha_windows_gl23x45",
    split="test",
    batch_size=8,
    num_workers=0,
    shuffle=False,
)
```

Pull one batch:

```python
batch = next(iter(train_loader))

tec_input = batch["tec_input"]
omni_input = batch["omni_input"]
target = batch["target"]
timestamp = batch["timestamp"]

print(tec_input.shape)    # torch.Size([8, 6, 23, 45])
print(omni_input.shape)   # torch.Size([8, 6, 6])
print(target.shape)       # torch.Size([8, 3, 23, 45])
print(timestamp.shape)    # torch.Size([8])
```

The model should output:

```python
prediction.shape == target.shape
```

That means:

```python
prediction.shape == [batch, 3, 23, 45]
```

## Direct Dataset Access

If a `DataLoader` is not needed:

```python
from data_pull.falisha_dataset import FalishaDTECDataset

ds = FalishaDTECDataset(
    root="data/falisha_windows_gl23x45",
    split="train",
)

sample = ds[0]

print(sample["tec_input"].shape)    # torch.Size([6, 23, 45])
print(sample["omni_input"].shape)   # torch.Size([6, 6])
print(sample["target"].shape)       # torch.Size([3, 23, 45])
print(sample["timestamp"])          # scalar epoch seconds
```

Available splits:

```python
split="train"
split="val"
split="test"
```

By default, `make_falisha_dataloader()` shuffles only the training split.

## Window Definition

Each sample contains:

```text
6 input dTEC frames
6 aligned driver frames
3 future target dTEC frames
```

Window assembly:

```python
tec_input[i]  = dtec[t : t + 6]
omni_input[i] = drivers[t : t + 6]
target[i]     = dtec[t + 6 : t + 9]
```

The builder rejects windows across:

```text
missing timestamps
cadence changes
duplicate timestamps
non-finite dTEC or driver values
```

## Validation

After unpacking the Google Drive archive, run this from the repository root:

```bash
python data_pull/falisha_dataset.py
```

Expected output shape summary:

```text
n=110124
{'tec_input': (6, 23, 45), 'omni_input': (6, 6), 'target': (3, 23, 45)}
timestamp=...
```

Batch validation:

```python
from data_pull.falisha_dataset import make_falisha_dataloader

loader = make_falisha_dataloader(
    root="data/falisha_windows_gl23x45",
    split="train",
    batch_size=8,
)

batch = next(iter(loader))

assert batch["tec_input"].shape == (8, 6, 23, 45)
assert batch["omni_input"].shape == (8, 6, 6)
assert batch["target"].shape == (8, 3, 23, 45)
```

Metadata validation:

```python
import json
from pathlib import Path

root = Path("data/falisha_windows_gl23x45")
meta = json.loads((root / "metadata.json").read_text())

assert meta["omni_features"] == [
    "b_magnitude",
    "by_gsm",
    "bz_gsm",
    "flow_speed",
    "proton_density",
    "kp_3hour",
]

print(meta["splits"])
print(meta["normalization"])
```

## Rebuilding The Dataset

Falisha normally does not need to rebuild the dataset. Rebuilding is only needed if changing preprocessing, date coverage, driver features, normalization, or window rules.

To rebuild the final windows from existing caches:

```bash
python data_pull/geomag_pull.py
python data_pull/data_for_falisha.py omni-cache --overwrite
python data_pull/data_for_falisha.py windows --overwrite
```

Full upstream rebuild:

```bash
python data_pull/data_interpolation.py --build
python data_pull/data_for_falisha.py iri-cache
python data_pull/data_for_falisha.py dtec-cache
python data_pull/geomag_pull.py
python data_pull/data_for_falisha.py omni-cache
python data_pull/data_for_falisha.py windows --overwrite
```

`geomag_pull.py` writes:

```text
data/raw/geomag/kp_daily.parquet
data/raw/geomag/kp_3hourly.parquet
```

`omni-cache` appends `kp_3hour` as the sixth driver channel.

## Repository Notes

The dataset folder is large and is delivered through Google Drive, not normal Git.

Keep this folder ignored by Git:

```text
data/falisha_windows_gl23x45/
```

The code files that belong in Git are:

```text
data_pull/data_interpolation.py
data_pull/data_time.py
data_pull/geomag_pull.py
data_pull/data_for_falisha.py
data_pull/falisha_dataset.py
```

