# README

## Description

This script implements a numerical solution for a 1D advection equation using the **Discontinuous Galerkin (DG) method**. The DG method is a finite element method that provides high-order accuracy and is effective for handling complex geometries. The script utilizes Lagrange interpolation polynomials and Gauss quadrature points to compute mass and stiffness matrices and applies an upwind flux scheme for stability. The simulation starts with an initial sine wave and evolves over a specified period.

## Prerequisites

- Python 3.7 or later
- Required Libraries: `numpy`, `matplotlib`

## Installation

To install the required libraries, run:

```bash
pip install numpy matplotlib