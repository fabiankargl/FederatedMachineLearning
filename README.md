# Flower Federated Learning Project

## Overview
This repository contains a complete experimental project for **Federated Learning using Flower (FL)**.  
It includes centralized benchmarks, federated baselines with different client configurations, and several experimental runs such as hyperparameter tuning and strategy evaluation.

The project is structured to separate core code (client/server/task) from results and baseline experiments to ensure clarity and reproducibility.

---

## Repository Structure

### `project/`
Contains the main source code of the project, including:
- **Client implementation**
- **Server implementation**
- **Task definitions** (model, training loop, evaluation logic)

This folder holds the actual runnable code for centralized and federated experiments.

---

## Baselines

### `baseline/`
Contains the **centralized baseline experiments**, including:
- Data analysis  
- Central model training  
- Evaluation results  
- Findings used to configure the federated baselines  

This serves as the reference performance for all FL experiments.

---

### `baseline_federated/`
Contains the **federated baseline experiments** using the best settings from the experiments.

Includes experiments with:
- **5, 15, and 41 clients**
- **IID** and **non-IID** data distributions
- comparing 2 different strategies

Each experiment folder stores results.

---

## Experiments

### `experiment/`
Contains results from additional experiments, such as:
- different values for **learning rate**, **local-epochs**, **number of trees**
- comparing 2 different strategies

This folder is intended for exploratory and extended experiments beyond the baseline.

---

## Additional Files

### `final_model.json`
Serialized model or configuration saved after finishing all experiments.

### `pyproject.toml`
Project configuration and dependencies.

### `.gitignore`
Git ignore rules for the project.

### `exploration_images/`
Contains plots of the data analysis

### `results_presentation/`
Contains results initially used for the presentation.

---
