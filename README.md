# Clothing Condition Classifier

Deep learning model to classify the physical condition of second-hand clothing items from images. Built for BYU CS 474 Final Project.

## Problem

Manual inspection of returned/second-hand clothing is a logistical bottleneck. This project trains a CNN to classify garment condition on a 1-5 scale using transfer learning.

## Dataset

[Second-Hand Fashion Dataset](https://fnauman.github.io/second-hand-fashion/) (Zenodo, 2024) — ~31,638 labeled clothing items (~30GB). Each item has front/back/brand images and JSON annotations including condition (1-5), pilling, stains, holes, usage, and more.

Collected by RISE Research Institutes of Sweden, Wargön Innovation AB, and Myrorna AB. Licensed CC-BY 4.0.

## Project Structure

```
├── data/                          # Dataset — not tracked (see .gitignore)
│   ├── clothing_v3/               #   Raw images + JSON labels (~30 GB)
│   └── cleaned_metadata_v2.csv    #   Processed CSV (output of data_cleaning.py)
├── src/                           # Python scripts
│   ├── data_cleaning.py           #   Raw JSON → cleaned CSV
│   ├── fraud_auditor.py           #   Train & evaluate vision-as-auditor models
│   ├── fraud_auditor_analysis.py  #   Post-hoc figures & tables for auditor
│   └── fraud_defect_from_vision.py #  V2 defect predictions → heuristic comparison
├── notebooks/                     # Jupyter notebooks (exploration & training)
│   ├── eda.ipynb                  #   Exploratory data analysis
│   ├── model.ipynb                #   V1 multi-task classifier
│   ├── model_v2.ipynb             #   V2 classifier (ordinal soft labels)
│   ├── model_regression.ipynb     #   Regression baseline
│   ├── error_analysis.ipynb       #   GradCAM & error breakdown
│   ├── station_experiment.ipynb   #   Per-station label quality analysis
│   ├── fraud_pipeline.ipynb       #   Fraud detection (embeddings + CLIP)
│   └── test.ipynb                 #   Scratch / quick tests
├── results/                       # Experiment writeups & output data
│   ├── Results.md                 #   V1 results
│   ├── Results_v2.md              #   V2 results
│   ├── Results_auditor.md         #   Vision-as-auditor results
│   ├── Results_error_analysis.md  #   Error analysis writeup
│   ├── Results_station_experiment.md # Station experiment writeup
│   └── *.csv, *.json              #   Model predictions & evaluation metrics
├── report/                        # Course deliverables
│   ├── Final Project Proposal.md
│   └── figures/                   #   All generated plots
├── checkpoints*/                  # Model weights — not tracked (see .gitignore)
├── autoresearch-master/           # Separate baseline project (original starter)
├── requirements.txt
├── LOGS.md
└── README.md
```
