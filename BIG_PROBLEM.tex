Here is a clean, comprehensive summary of the project framework. You can copy and paste this text or save it directly into a markdown file (e.g., project_proposal.md) to share with your friends.
------------------------------
## Project Proposal: S2Kp-Net
A Geometry-Aware Spherical CNN for Forecasting Global Kp Index from Sparse Geomagnetic Observatories
## 1. The Core Problem & Our Pitch
The Kp index (Planetary K-index) is a single scalar value (0–9) that summarizes global geomagnetic activity over a 3-hour window. Standard space-weather models use flat time-series arrays or flat 2D maps to forecast it.
## The Flaw in Existing Models
Geomagnetic activity (like auroras) concentrates heavily around the North and South magnetic poles.

* Flat 2D maps (like Mercator projections) introduce severe spatial distortion at the poles.
* Standard 2D CNN kernels get warped, miscalculating physical boundaries.
* Flat time-series models (like LSTMs) treat sensors as a generic list, completely ignoring Earth's 3D geometry and rotation.

## Our Solution
We will use a Spherical CNN ($S^2\text{CNN}$) to process ground magnetometer data directly on a 3D spherical manifold ($S^2$). This allows our model to learn the spatial relationships of global magnetic storms natively, without polar distortion or coordinate artifacts.
------------------------------
## 2. Data & Input Framework (Moving Beyond the Single Number)
To make an $S^2\text{CNN}$ useful for Kp prediction, we cannot just look at past Kp values. We must look at the spatial data that creates the Kp index.

       [ 13 Ground Stations ] ----> Map to Lat/Lon Coordinates
                                              │
                                              ▼
       [ Blank Spherical Canvas ] -> Drop ΔB values into exact pixels
                                              │
                                              ▼
       [ S2CNN Input Layer ] -------> Continuous 3D Earth Globe


   1. The Source: Download historical time-series data from the 13 specific geomagnetic observatories used to calculate the Kp index (available via the Intermagnet Network or NOAA NCEI).
   2. The Spherical Canvas: For every 3-hour window, initialize a blank 3D sphere grid using HEALPix or an equiangular grid.
   3. Sparse Mapping: Place the 13 observatories at their exact latitude/longitude coordinates on the sphere. Populate those pixels with their recorded magnetic perturbation values ($\Delta B$). Leave the rest of the sphere blank (zero-filled).
   4. Target Output: The continuous scalar Kp index (0–9) for the next 3 to 6 hours.

------------------------------
## 3. Network Architecture Design ($S^2 \to SO(3) \to \text{Scalar}$)
Our pipeline respects the true physical and geometric properties of a rotating Earth:

[ Input: S² Sphere ] 
        │
        ▼  (S² Convolutions via Spherical Harmonics)
[ Hidden: SO(3) Feature Maps ]  <-- Rotational Equivariance Encoded
        │
        ▼  (Global Average Pooling)
[ 1D Feature Vector ]
        │
        ▼  (Linear Fully Connected Layer)
[ Output: Continuous Scalar Kp ]


* Layer 1 ($S^2 \to SO(3)$): Convolves the sparse input sphere with filters using Spherical Harmonics. This transforms the 2D sphere into a 3D space of rotations ($SO(3)$).
* Layer 2 ($SO(3) \to SO(3)$): Processes features directly in the rotational manifold.
* Global Pooling: Collapses the spatial dimensions, leaving a dense vector of the most critical learned features.
* Output Head: A standard linear layer mapping the feature vector to a single predicted Kp value.

------------------------------
## 4. Key Scientific Arguments for a Paper
To make this a publishable project, our text will focus on three core machine learning/physics principles:
## A. Resolution of Polar Distortion
By performing calculus directly on the surface of the sphere, the area element adjusts natively ($du = \sin\theta d\theta d\phi$). The model processes space storms near the poles with perfect mathematical fidelity.
## B. Rotational Equivariance ($SO(3)$ Symmetry)
As the Earth rotates relative to the sun, a magnetic storm pattern hitting Canada at noon looks physically identical to a storm hitting Siberia hours later. Standard models have to memorize these variations independently. Because our $S^2\text{CNN}$ is inherently equivariant to 3D rotations, it automatically generalizes this behavior across the entire globe without extra training data.
## C. Explainable AI via Spatial Heatmaps
By applying feature visualization techniques (like Grad-CAM) to our trained model, we can project a heatmap back onto the Earth. This shows geophysicists exactly which regions of the magnetosphere were most mathematically responsible for driving a major global storm event.
------------------------------
## 5. Implementation Action Plan

* Step 1: Download the Kp index labels along with the raw 3-component magnetic data for the 13 Intermagnet stations.
* Step 2: Write a script to pixelate this data onto a sphere using the healpy or e2cnn library.
* Step 3: Build a naive Baseline Model (a basic linear model or a standard flat 2D CNN) to benchmark against.
* Step 4: Deploy the Spherical CNN pipeline and compare the Mean Squared Error (MSE) against the baselines to prove that geometry matters.

------------------------------
How do your friends feel about handling the data collection side, or should we look into a code template to get the sphere initialized first?

