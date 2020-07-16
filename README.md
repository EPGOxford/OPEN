# Open Platform for Energy Networks (OPEN)

## Overview

Oxford University's Energy and Power Group's Open Platform for Energy Networks (OPEN) provides a python toolset for modelling, simulation and optimisation of smart local energy systems.
The framework combines distributed energy resource modelling (e.g. for PV generation sources, battery energy storage systems, electric vehicles), energy market modelling, power flow simulation and multi-period optimisation for scheduling flexible energy resources.

OPEN and the methods used are presented in detail in the following publication:

T. Morstyn, K. Collett, A. Vijay, M. Deakin, S. Wheeler, S. M. Bhagavathy, F. Fele and M. D. McCulloch; *"An Open-Source Platform for Developing Smart Local Energy System Applications”*; University of Oxford Working Paper, 2019

## Documentation

The full OPEN documentation can be found [here](https://open-platform-for-energy-networks.readthedocs.io).

## Installation

Download OPEN source code.

If using conda, we suggest creating a new virtual environment from the requirements.txt file.
First, add the following channels to your conda distribution if not already present:

    conda config --add channels invenia
    conda config --add channels picos
    conda config --add channels conda-forge

To create the new virtual environment, run:

    conda create --name <env_name> --file requirements.txt python=3.6

## Getting started

The simplest way to start is to duplicate one of the case study main.py files:
- OxEMF_EV_case_study_v6.py
- Main_building_casestudy.py


## Contributors

* Thomas Morstyn
* Avinash Vijay
* Katherine Collet
* Filiberto Fele
* Matthew Deakin
* Sivapriya Mothilal Bhagavathy
* Scot Wheeler
* Malcolm McCulloch
