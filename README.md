# MNIST Adversarial Robustness Project

## What I have done so far

This project is focused on building and testing an MNIST image classifier under adversarial attacks.

- Implemented a CNN-based MNIST model in the model module.
- Added a FastAPI endpoint for image prediction and health checks.
- Set up the project to load a trained model from the models directory and serve predictions locally.
- Containerized the app using Docker so it can run consistently in a reproducible environment.
- Added adversarial attack scripts for FGSM and PGD style evaluation.
- Created a basic attack pipeline and reporting structure for tracking robustness experiments.
- Included dataset and adversarial example folders to support testing and comparison work.

## Current status

The project is in an early but functional stage: the model, API, Docker setup, and attack pipeline are in place, and the next step is to run and compare adversarial evaluations more systematically.

## Planned next steps

- Run attack experiments across multiple epsilon values.
- Compare clean vs. adversarial model performance.
- Capture metrics and screenshots for the report.
- Refine the API and evaluation workflow for clearer results.
