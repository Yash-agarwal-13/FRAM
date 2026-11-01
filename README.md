# Financial Risk Analytics and Management (FRAM)
## Credit Risk Modelling — Biocon Limited

### Project Overview

This repository contains the Python implementation and Jupyter notebooks developed for the Financial Risk Analytics and Management (FRAM) course assignment.

**Company:** Biocon Limited  
**Programming Language:** Python 3.12  
**Development Environment:** Visual Studio Code  
**Version Control:** Git and GitHub  
**Final Deliverable:** Consolidated Jupyter Notebook (`.ipynb`), executable in Google Colab

### Project Objectives

The project aims to apply quantitative financial risk modelling techniques to Biocon Limited, including:

- Black-Scholes-Merton (BSM) framework
- Merton structural credit risk model
- GARCH volatility modelling
- Machine learning techniques for financial risk analysis
- Model evaluation, interpretation and comparison

The detailed implementation and evaluation methodology will follow the course assignment requirements.

### Project Structure

```text
FRAM_BIOCON/
├── data/               # Input datasets
├── src/                # Python source code
├── notebooks/          # Independent Jupyter notebooks
├── outputs/            # Generated figures and results
├── final/              # Final consolidated notebook
├── .gitignore          # Git exclusions
├── README.md           # Project documentation
└── requirements.txt    # Python dependencies
```

### Development Workflow

The project follows a modular development approach:

1. Develop and test individual models independently.
2. Track changes using Git feature branches.
3. Merge validated work into the `develop` branch.
4. Integrate and test the complete workflow.
5. Merge the validated version into `main`.
6. Prepare the final Jupyter notebook for Google Colab.

### Environment Setup

Create a Python virtual environment:

```powershell
py -m venv .venv
```

Activate it in Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

### Project Status

**Current phase:** Development environment and repository setup.

Financial modelling and analysis are pending implementation.

### Academic Use

Developed as part of a university course assignment. Results will be based on the specified financial data and modelling assumptions and should not be interpreted as investment recommendations.python -m pip install --upgrade pip