# Thermal Nueral Operator (ThermaNO): Scalable Scientific Machine Learning for Additive Manufacturing

### Project Overview
This project develops both physics-informed and data-driven neural operator architectures for solving the 3D transient heat equation in Laser Powder Bed Fusion (LPBF).

LPBF involves highly nonlinear heat transfer, moving heat sources, and multi-scale spatio-temporal dynamics. These characteristics make traditional numerical solvers (e.g., FDM/FEM) computationally expensive — particularly for process optimization tasks, where hundreds of scan scenarios must be simulated.

ThermaNO leverages state-of-the-art Scientific Machine Learning (SciML) architectures, DeepONet-based neural operators, to deliver fast, scalable, and reliable thermal predictions.

The framework enables:

-  Accurate melt pool geometry prediction
- Full-field temperature distribution estimation
- Single-track and multi-track scan path modeling
- Rapid parametric studies for process optimization

### Dual Learning Modes
ThermaNO supports two learning paradigms:

:one: Physics-Informed Variant (Unsupervised)
- Enforces the governing heat equation directly in the loss function
- Requires no labeled training data
- Ensures physical consistency
- Suitable when simulation data is limited or expensive

The results of the physics-informed DeepONet are available on [arXiv](https://arxiv.org/abs/2502.01820).

:two: Data-Driven Variant (Supervised)

- Trained on a small number of temperature snapshots
- Learns the operator mapping between scan path and temperature field
- Generalizes to unseen time steps
- Suitable for multi-track or multi-material configurations where physics-informed variant loses accuracy due to the curse of dimensionality.

A manuscript on the optimization of multi-material LPBF using the data-driven ThermaNO framework is currently in preparation.

