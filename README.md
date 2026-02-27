# Clothing Condition Classifier

Deep learning model to classify the physical condition of second-hand clothing items from images. Built for BYU CS 474 Final Project.

## Problem

Manual inspection of returned/second-hand clothing is a logistical bottleneck. This project trains a CNN to classify garment condition on a 1-5 scale using transfer learning.

## Dataset

[Second-Hand Fashion Dataset](https://fnauman.github.io/second-hand-fashion/) (Zenodo, 2024) — ~31,638 labeled clothing items (~30GB). Each item has front/back/brand images and JSON annotations including condition (1-5), pilling, stains, holes, usage, and more.

Collected by RISE Research Institutes of Sweden, Wargön Innovation AB, and Myrorna AB. Licensed CC-BY 4.0.

## Project Structure

```
├── data/clothing_v3/   # Dataset (not tracked in git)
├── report/             # Proposal and final report
├── src/                # Source code and notebooks
└── requirements.txt
```
