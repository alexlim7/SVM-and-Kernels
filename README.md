# SVM and Kernels Visualization

This project explores **Support Vector Machines (SVMs)**, kernel methods, and higher-dimensional embeddings. It provides 2D and 3D visualizations of decision boundaries, margins, and support vectors for both linear and non-linear data.

## Features

- **SVM Training**: Supports linear, RBF, and custom polar kernels.  
- **Kernel Embeddings**: Maps 2D data to 3D using polar embedding.  
- **Decision Boundary Visualization**: Shows SVM hyperplanes in original and embedded spaces.  
- **Support Vector Visualization**: Plots margins and highlights support vectors.  
- **Slack Penalty Exploration**: Demonstrates the effect of different `C` values on RBF SVM boundaries.  

## How to Run

The main entry point is `main.py`. Running it will generate all visualizations:

```bash
python main.py
```

The script will:
- Load datasets from the data/ folder.
- Train SVMs using linear, RBF, and polar kernels.
- Plot decision boundaries in 2D and 3D.
- Visualize H(x) and support vectors.

## Project Structure
```text
.
├── data/                              # Folder containing all dataset CSV files
├── main.py                            # Main script to run all visualizations
├── utils.py                           # Utility functions and visualizer class
├── support_vector_visualization.py    # Functions to plot support vectors and margins
├── decision_boundary_visualization.py # Polar kernel, embedding, and H(x) visualization
└── README.md                          # Project documentation
