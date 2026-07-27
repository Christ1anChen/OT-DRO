# OT-DRO: Oracle-based Distributionally Robust Optimization under Optimal Transport Ambiguity Sets

## Overview

This repository contains the official Python implementation of a scalable computational framework for Optimal Transport Distributionally Robust Optimization (OT-DRO). It provides efficient algorithms designed to solve min-max DRO problems involving piecewise quadratic loss functions and an $\ell_2$ transportation cost. 

The core contributions implemented in this repository include:
*   **Inner Worst-Case Expectation Oracle:** A highly efficient, inherently parallelizable budget allocation algorithm that avoids the dense matrix lifting bottlenecks of standard exact commercial solvers.
*   **Distributional Best-Response (DBR):** An online gradient-based best-response algorithm for solving the primal Min-Max DRO problem over the bounded feasible region.
*   **Support Compression Algorithms:** Post-processing methods to ensure the sparsity of the worst-case/least-favorable distributions. This includes primal sorting-based greedy heuristics, as well as exact restricted dual and relaxed tangent-based compression programs.

---

## Repository Structure

The project is structured as a local Python package (`OT-DRO`) with a standard `src` layout.

```text
OT-DRO/
├── src/
│   ├── inner/                  # Solvers for the inner worst-case expectation problem
│   │   ├── __init__.py
│   │   ├── compression.py      # Sorting-based greedy support compression
│   │   ├── losses.py           # Formulation of K-piece convex-concave saddle point losses
│   │   ├── solver_l2.py        # Core budget allocation algorithm implementation
│   │   ├── solver_l2_gurobi.py # SOCP exact baseline using Gurobi
│   │   └── solver_l2_mosek.py  # SOCP exact baseline using MOSEK
│   │
│   └── primal/                 # Solvers for the outer/primal DRO problem
│       ├── __init__.py
│       ├── compression_dual_l2.py    # Exact restricted dual compression program
│       ├── compression_tangent_l2.py # Relaxed tangent-based compression linear program
│       ├── dbr_solver_l2.py          # Distributional Best-Response (DBR) algorithm
│       └── full_dual_l2.py           # Full dual baseline formulation for exact optimum
│
├── test/                       # Execution scripts and synthetic numerical experiments
│   ├── test_dbr_l2.py          # Primal Min-Max DRO experiments (Section 6.3)
│   └── test_inner_solver_l2.py # Scalability benchmarking for inner oracle (Section 6.2)
│
├── .gitignore                  # Git tracking exclusions (e.g., __pycache__)
├── LICENSE                     # Software license
├── requirements.txt            # Conda environment dependencies
└── pyproject.toml              # Build system and package dependency configurations
```

---

## Prerequisites & Installation

### Requirements
* Python 3.8+
* `numpy`
* `scipy`
* **Commercial Solver Licenses:** To run the baseline exact SOCP solvers, you must have active licenses for **MOSEK** (configured via `mosek.lic`) and **Gurobi** (configured via `grbgetkey`).

### Installation
To utilize the interconnected folder structure, clone the repository and install it in editable mode using the provided `pyproject.toml`.

```bash
# Clone the repository
git clone [https://github.com/Christ1anChen/OT-DRO.git](https://github.com/Christ1anChen/OT-DRO.git)
cd OT-DRO

# Create and activate a virtual environment
conda create -n otdro python=3.11
conda activate otdro

# Install the package and dependencies
conda install --yes --file requirements.txt
pip install -e .
```

---

## Usage
Once installed, you can execute the numerical experiments directly from the root directory. The test scripts dynamically generate synthetic datasets consisting of random multivariate normal samples and piecewise quadratic loss components, executing the solver routines and tracking the performance metrics.

### Evaluating the Inner Worst-Case Solver
To benchmark the budget allocation algorithm against MOSEK and Gurobi:

```Bash
python test/test_inner_solver_l2.py
```

### Running the Distributional Best-Response (DBR) Algorithm
To run the DBR algorithm for primal DRO along with the dual and tangent compression steps:

```Bash
python test/test_dbr_l2.py
```
