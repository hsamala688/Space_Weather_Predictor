# TEC-S2Net — Visualization Deliverables

Two visuals, split between Falisha (offline renders for the paper +
an MP4) and Hayden (interactive web globe). They share the same data
export format so neither blocks the other.

---

## Data format (the handoff)

Everything below assumes model output as a plain NumPy array on the
native IONEX grid:

```
tec_pred   : (T, 71, 73)   float32   predicted TEC (or ΔTEC + IRI)
tec_true   : (T, 71, 73)   float32   ground truth, same timestamps
lats       : (71,)         float32   +87.5 → −87.5   (descending)
lons       : (73,)         float32   −180 → +180
timestamps : (T,)          int64     unix seconds
```

Save as a single `.npz`. That one file feeds both deliverables.

**Important:** these are absolute TEC in TECU, not ΔTEC. Add the IRI
baseline back before exporting — a globe of residuals is not what a
reader wants to look at.

---

# PART 1 — Falisha: PyVista globe renders

## Why this is easy now

We are on a regular 2.5° × 5° lat/lon grid, so PyVista's
`grid_from_sph_coords` builds the sphere mesh directly from our
coordinate arrays. No interpolation, no triangulation, no HEALPix
pixel conversion.

## Install

```bash
pip install pyvista imageio imageio-ffmpeg
```

## Template: single frame (do this first)

Get one frame looking correct before touching the animation loop.

```python
import numpy as np
import pyvista as pv

d          = np.load("tec_frames.npz")
tec_pred   = d["tec_pred"]      # (T, 71, 73)
lats       = d["lats"]          # +87.5 → −87.5
lons       = d["lons"]          # −180 → +180

# --- build the sphere mesh once -------------------------------
# grid_from_sph_coords wants: azimuth (0–360), polar angle (0–180), radius
theta = (lons + 180.0) % 360.0        # longitude  → azimuth  [0, 360)
phi   = 90.0 - lats                   # latitude   → colatitude [0, 180]
grid  = pv.grid_from_sph_coords(theta, phi, np.array([1.0]))

# --- attach one frame of scalars ------------------------------
frame = tec_pred[0]                   # (71, 73)
grid.point_data["TEC"] = frame.ravel(order="F")   # ← see ORDERING NOTE

p = pv.Plotter(window_size=(900, 900))
p.add_mesh(
    grid,
    scalars="TEC",
    cmap="plasma",
    clim=[0, 100],                    # fix the scale — see COLOR SCALE
    smooth_shading=True,
    scalar_bar_args={"title": "TEC (TECU)"},
)
p.add_axes()
p.show()
```

### ORDERING NOTE — the one real gotcha

`ravel(order="F")` vs `ravel(order="C")` and `lats` ascending vs
descending are the only things that can silently produce a
wrong-but-plausible globe. Validate once, explicitly:

1. Pick a frame at a known UT — say 12:00 UT.
2. Render it.
3. The dayside ionization bulge (the big bright blob) must sit
   over the longitude where local noon is at that UT. At 12:00 UT
   that is roughly the prime meridian, 0° longitude.

If the bulge is on the wrong side, flip the longitude convention.
If it is mirrored north/south, reverse `lats` and the array rows
together (`lats[::-1]` and `frame[::-1]`). If it looks scrambled or
striped, switch `order="F"` ↔ `order="C"`.

Do not proceed to the animation until one static frame is verifiably
correct.

### COLOR SCALE

Fix `clim` across every frame. If PyVista auto-scales per frame, the
colorbar changes underneath the animation and a storm looks identical
to a quiet day. Compute it once from the whole array:

```python
vmin = 0.0
vmax = float(np.percentile(np.concatenate([tec_true, tec_pred]), 99))
```

Percentile rather than max, so one extreme pixel doesn't wash out the
scale.

## Template: side-by-side storm animation (the deliverable)

```python
import numpy as np
import pyvista as pv

d          = np.load("tec_frames.npz")
tec_pred   = d["tec_pred"]
tec_true   = d["tec_true"]
lats, lons = d["lats"], d["lons"]

theta = (lons + 180.0) % 360.0
phi   = 90.0 - lats

grid_t = pv.grid_from_sph_coords(theta, phi, np.array([1.0]))
grid_p = pv.grid_from_sph_coords(theta, phi, np.array([1.0]))

vmin = 0.0
vmax = float(np.percentile(np.concatenate([tec_true, tec_pred]), 99))

p = pv.Plotter(shape=(1, 2), window_size=(1600, 800), off_screen=True)
p.open_movie("storm_forecast.mp4", framerate=8)

# left: ground truth
p.subplot(0, 0)
grid_t.point_data["TEC"] = tec_true[0].ravel(order="F")
p.add_mesh(grid_t, scalars="TEC", cmap="plasma", clim=[vmin, vmax],
           smooth_shading=True, scalar_bar_args={"title": "TEC (TECU)"})
p.add_text("Observed (IONEX)", font_size=14)

# right: prediction
p.subplot(0, 1)
grid_p.point_data["TEC"] = tec_pred[0].ravel(order="F")
p.add_mesh(grid_p, scalars="TEC", cmap="plasma", clim=[vmin, vmax],
           smooth_shading=True, scalar_bar_args={"title": "TEC (TECU)"})
p.add_text("TEC-S2Net forecast", font_size=14)

p.link_views()          # both globes rotate together
p.camera_position = "xy"

for t in range(len(tec_pred)):
    grid_t.point_data["TEC"] = tec_true[t].ravel(order="F")
    grid_p.point_data["TEC"] = tec_pred[t].ravel(order="F")
    p.camera.azimuth = 0.5 * t        # slow drift; drop if distracting
    p.write_frame()

p.close()
print("wrote storm_forecast.mp4")
```

Run this on a storm window from the **test** set — St. Patrick's Day
2015 (17–18 March) is the canonical event and is in our test range if
the split allows; otherwise pick the largest Kp event in test.

`off_screen=True` matters: it renders without opening a window, which
is what you want on a headless machine or in a script.

## Optional third panel: error

Same pattern with `shape=(1, 3)`, third grid holding
`tec_pred[t] - tec_true[t]`, using a diverging colormap
(`cmap="RdBu_r"`) and a symmetric `clim=[-e, +e]`. Diverging + symmetric
is non-negotiable for signed error — otherwise zero isn't at the
center of the colormap and over/under-prediction look the same.

## Deliverables from Part 1

| File | Use |
|---|---|
| `storm_forecast.mp4` | README, presentations, demo |
| `frame_peak.png` | static figure for the paper |
| `frame_error.png` | error panel, optional paper figure |

---

# PART 2 — Hayden: interactive web globe

## What this is

Falisha's MP4 is fixed — one camera path, one event. The web version
lets someone rotate the globe themselves and scrub a time slider. It
is the thing to put in the README and open during a presentation.

Best fit for React: **`react-globe.gl`**. It wraps three.js and maps
an **equirectangular image** onto a sphere natively — and an
equirectangular image is exactly what our 71×73 lat/lon grid is. So
the data pipeline is: array → PNG → texture. No geometry work at all.

## Step 1 — export frames as PNG textures (Python, one-time)

```python
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import json, os

d = np.load("tec_frames.npz")
tec, ts = d["tec_pred"], d["timestamps"]

os.makedirs("public/frames", exist_ok=True)

vmin, vmax = 0.0, float(np.percentile(tec, 99))
norm = colors.Normalize(vmin=vmin, vmax=vmax)
cmap = cm.get_cmap("plasma")

for i, frame in enumerate(tec):
    rgba = cmap(norm(frame))                  # (71, 73, 4) in [0,1]
    plt.imsave(f"public/frames/tec_{i:04d}.png", rgba)

meta = {
    "n_frames": len(tec),
    "vmin": vmin,
    "vmax": vmax,
    "timestamps": [int(t) for t in ts],
}
with open("public/frames/meta.json", "w") as f:
    json.dump(meta, f)

print(f"wrote {len(tec)} frames")
```

Row order check: if the globe comes out upside down, the array rows
are descending in latitude while equirectangular textures expect north
at the top row — flip with `frame[::-1]` before `imsave`.

## Step 2 — React component

```bash
npm install react-globe.gl
```

```jsx
import { useState, useEffect } from "react";
import Globe from "react-globe.gl";

export default function TECGlobe() {
  const [meta, setMeta]   = useState(null);
  const [frame, setFrame] = useState(0);
  const [playing, setPlaying] = useState(false);

  useEffect(() => {
    fetch("/frames/meta.json").then(r => r.json()).then(setMeta);
  }, []);

  useEffect(() => {
    if (!playing || !meta) return;
    const id = setInterval(
      () => setFrame(f => (f + 1) % meta.n_frames),
      125                                   // 8 fps
    );
    return () => clearInterval(id);
  }, [playing, meta]);

  if (!meta) return <div>loading…</div>;

  const pad = String(frame).padStart(4, "0");
  const when = new Date(meta.timestamps[frame] * 1000)
    .toISOString().replace("T", " ").slice(0, 16) + " UT";

  return (
    <div>
      <Globe
        globeImageUrl={`/frames/tec_${pad}.png`}
        backgroundColor="#000010"
        showAtmosphere={true}
        width={800}
        height={800}
      />
      <div style={{ padding: 12, color: "#eee", fontFamily: "monospace" }}>
        <button onClick={() => setPlaying(p => !p)}>
          {playing ? "pause" : "play"}
        </button>
        <input
          type="range"
          min={0}
          max={meta.n_frames - 1}
          value={frame}
          onChange={e => setFrame(Number(e.target.value))}
          style={{ width: 400, marginLeft: 12 }}
        />
        <span style={{ marginLeft: 12 }}>{when}</span>
      </div>
    </div>
  );
}
```

That is the whole thing. `globeImageUrl` swaps the texture on each
frame change; three.js handles the sphere, rotation, and zoom.

## Step 3 — polish, in priority order

1. **Coastline overlay.** `react-globe.gl` accepts a
   `polygonsData` prop; feed it a countries GeoJSON so the TEC field
   is geographically readable. Without this the bulge is a pretty blob
   with no reference frame.
2. **Colorbar.** A static PNG or CSS gradient labelled with `vmin`/
   `vmax` from `meta.json`. A heatmap without a scale is decoration,
   not data.
3. **Observed / forecast toggle.** Export a second frame directory
   from `tec_true` and switch `globeImageUrl` between them — makes the
   comparison interactive.
4. **Preload.** Fetch the next few frames into `Image()` objects so
   playback doesn't stutter on first pass.

## Scope note

Keep the frame count modest — a 48-hour storm window at hourly cadence
is 48 PNGs, a few MB total, which loads instantly. Do not export the
whole test set.

---

# Division of labour

| | Falisha | Hayden |
|---|---|---|
| Owns | PyVista renders, MP4, paper figures | React globe, time scrubber, deploy |
| Input | `tec_frames.npz` | `public/frames/*.png` + `meta.json` |
| Output | `storm_forecast.mp4`, static PNGs | interactive page |
| Blocking on | trained model producing forecasts | the PNG export script (Part 2 Step 1) |

Neither is blocked by the other once `tec_frames.npz` exists. The PNG
export script is ~20 lines and can be run by whoever gets there first.

---

# Validation checklist

Before either visual is considered done:

- [ ] Dayside bulge sits at the correct longitude for the frame's UT
- [ ] North is up (Arctic at top, Antarctic at bottom)
- [ ] Colour scale is fixed across all frames, not auto-ranged
- [ ] Colorbar is present and labelled in TECU
- [ ] Storm frames visibly differ from quiet frames
- [ ] Observed and forecast panels share one colour scale
- [ ] Error panel (if used) is diverging and symmetric about zero
