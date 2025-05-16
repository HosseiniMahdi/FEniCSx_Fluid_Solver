## FEniCSx Based Fluid Solver

This repository contains the code developed as part of my master's thesis, focused on implementing a finite element solver for fluid flow using the [FEniCSx](https://fenicsproject.org/) framework. The goal is to solve Stokes equations and to investigate the accuracy of higher-order elements.

## Installation
With conda FEniCSx is installed with:

```
conda create -n fenicsx-env
conda activate fenicsx-env
conda install -c conda-forge fenics-dolfinx mpich pyvista
```
On Debian or Ubuntu it is installed with:

```
sudo add-apt-repository ppa:fenics-packages/fenics
sudo apt update
sudo apt install fenicsx
```

## Simulations

The python scripts for different setups are located in the `src` folder.
