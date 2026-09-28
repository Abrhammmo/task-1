# AI Security Red Team Assessment

A local AI security red-team assessment of a PyTorch MNIST image-classification model and its Dockerized FastAPI serving pipeline.

The assessment evaluates both **ML-level adversarial attacks** and **system-level weaknesses** in the model-serving pipeline.

> **Scope:** All attacks and security tests were performed against the locally controlled model and Docker container. No external systems were targeted.

---

## 1. Assessment Overview

The target system consists of:

```text
                    Attacker
                       |
                       v
                FastAPI /predict
                       |
                       v
                Docker Container
                       |
                       v
                  MNIST CNN
                       |
                       v
             Class + Confidence
```

The assessment covers:

* Baseline model evaluation
* FGSM adversarial attacks
* PGD adversarial attacks
* API-level adversarial validation
* File-upload and input-validation testing
* Authentication testing
* Rate-limiting testing
* Docker privilege assessment
* Docker resource-limit assessment
* Model artifact and loading review
* Security hardening recommendations
* MITRE ATLAS mapping
* NIST AI RMF mapping

---

## 2. Objectives

The assessment was designed to:

1. Analyze and attack a local ML model.
2. Evaluate the model-serving pipeline.
3. Create successful adversarial inputs.
4. Identify system-level security weaknesses.
5. Assess potential metadata and information-disclosure risks.
6. Propose realistic hardening recommendations.
7. Map relevant findings to MITRE ATLAS and NIST AI RMF.

---

## 3. Technology Stack

| Component           | Technology                          |
| ------------------- | ----------------------------------- |
| Language            | Python 3.11                         |
| ML Framework        | PyTorch                             |
| Dataset             | MNIST                               |
| Model               | Custom CNN                          |
| API                 | FastAPI                             |
| Image Processing    | Pillow                              |
| Containerization    | Docker                              |
| Adversarial Attacks | Custom FGSM and PGD implementations |
| Runtime             | CPU                                 |
| Testing             | PowerShell / Python                 |
| Documentation       | Markdown                            |

---

## 4. Project Structure

```text
task-1/
│
├── models/
│   └── mnist_cnn.pth
│
├── data/
│
├── src/
│   ├── model/
│   │   ├── __init__.py
│   │   ├── model.py
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── predict.py
│   │
│   ├── attacks/
│   │   ├── __init__.py
│   │   ├── fgsm.py
│   │   ├── fgsm_search.py
│   │   ├── visualize_fgsm.py
│   │   ├── pgd.py
│   │   ├── pgd_search.py
│   │   └── save_pgd_images.py
│   │
│   └── api/
│       └── main.py
│
├── adversarial_examples/
│   ├── fgsm/
│   └── pgd/
│
├── reports/
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── requirements-api.txt
└── README.md
```

---

# 5. Model

The target is a small convolutional neural network trained on MNIST.

Architecture:

```text
Input: 1 × 28 × 28
        |
Conv2D: 1 → 32
        |
      ReLU
        |
   MaxPool 2×2
        |
Conv2D: 32 → 64
        |
      ReLU
        |
   MaxPool 2×2
        |
      Flatten
        |
Linear: 64×7×7 → 128
        |
      ReLU
        |
Linear: 128 → 10
        |
   Class logits
```

The model was selected to provide a lightweight but meaningful image-classification target suitable for local CPU-based adversarial testing.

---

# 6. Baseline Results

The trained model achieved:

```text
Test samples:       10,000
Correct predictions: 9,879
Accuracy:             98.79%
```

This establishes the baseline against which adversarial behavior was evaluated.

Only samples that were initially classified correctly were counted when calculating adversarial attack success. This prevents pre-existing model errors from being incorrectly attributed to the attack.

---

# 7. FGSM Attack

Fast Gradient Sign Method (FGSM) was implemented as a one-step white-box attack:

```python
adversarial_image = image + epsilon * gradient.sign()
```

The attack modifies the input in the direction that increases the classification loss.

### FGSM Results

The first 1,000 MNIST test samples were evaluated. Of these, 988 were initially classified correctly.

| Epsilon | Successful Attacks | Success Rate |
| ------: | -----------------: | -----------: |
|    0.10 |                 36 |        3.64% |
|    0.20 |                126 |       12.75% |
|    0.30 |                275 |       27.83% |
|    0.40 |                457 |       46.26% |
|    0.50 |                599 |       60.63% |
|    0.60 |                714 |       72.27% |
|    0.70 |                791 |       80.06% |
|    0.80 |                856 |       86.64% |

The results show increasing attack success as the perturbation budget increases.

### Example

At epsilon `0.10`:

```text
True label:          5
Clean prediction:    5
Clean confidence:    52.79%

Adversarial prediction: 8
Adversarial confidence: 91.49%
```

The adversarial example is available under:

```text
adversarial_examples/fgsm/epsilon_0.10/
```

---

# 8. PGD Attack

Projected Gradient Descent (PGD) was implemented as an iterative white-box attack.

Configuration:

```text
Alpha: 0.01
Steps: 40
```

The attack performs repeated gradient updates and projects the resulting perturbation back into the allowed epsilon range.

### PGD Results

The first 1,000 test images were evaluated, with 988 initially correct samples.

| Epsilon | Successful Attacks | Success Rate |
| ------: | -----------------: | -----------: |
|    0.05 |                 23 |        2.33% |
|    0.10 |                 48 |        4.86% |
|    0.15 |                120 |       12.15% |
|    0.20 |                247 |       25.00% |
|    0.25 |                393 |       39.78% |
|    0.30 |                553 |       55.97% |

The attack success rate increased as the perturbation budget increased.

---

# 9. End-to-End API Validation

White-box success against a tensor does not necessarily guarantee success against the deployed API.

Adversarial tensors were therefore converted into image files and submitted through the actual FastAPI pipeline.

The API performs:

```text
Image upload
     ↓
Pillow decoding
     ↓
Grayscale conversion
     ↓
Resize to 28×28
     ↓
Tensor conversion
     ↓
Normalization
     ↓
Model inference
```

Six saved PGD examples were tested through the API.

Two of the six remained successful after image serialization and API preprocessing.

### Demonstrated API-Level Attack

One successful example produced:

```json
{
  "class": 8,
  "confidence": 0.7007153034210205
}
```

The corresponding clean input was classified as:

```json
{
  "class": 5,
  "confidence": 0.52828449010849
}
```

This demonstrates that an adversarial input successfully crossed the model-serving pipeline and caused a misclassification.

The fact that some PGD examples did not survive the API pipeline was treated as an experimental observation rather than ignored. Image serialization and preprocessing can alter the exact perturbation.

---

# 10. FastAPI Service

The API exposes two endpoints:

### Health Check

```http
GET /health
```

Example:

```json
{
  "status": "ok",
  "device": "cpu"
}
```

### Prediction

```http
POST /predict
```

The endpoint accepts an uploaded image and returns:

```json
{
  "class": 8,
  "confidence": 0.7007153034210205
}
```

---

# 11. Security Findings

The assessment considered both ML-specific and conventional application/container security.

## 11.1 Successful adversarial misclassification

**Status:** Demonstrated

Adversarial inputs generated using FGSM and PGD caused model misclassification. At least one adversarial example was also successfully validated through the Dockerized API.

**Impact:**

An attacker capable of constructing suitable adversarial inputs may influence the model's classification result.

---

## 11.2 Unauthenticated inference endpoint

**Status:** Demonstrated

The `/predict` endpoint accepts inference requests without authentication.

**Impact:**

Any party able to reach the service can submit inference requests.

**Recommended hardening:**

* Authentication
* Authorization
* API keys or tokens where appropriate
* Network access controls

---

## 11.3 No observed rate limiting

**Status:** Demonstrated under controlled testing

Repeated inference requests were submitted without authentication.

No `429 Too Many Requests` response was observed during the controlled test.

**Recommended hardening:**

* Rate limiting
* Request quotas
* Authentication
* Monitoring and alerting

---

## 11.4 Insufficient invalid-input handling

**Status:** Demonstrated

An invalid uploaded file resulted in:

```text
HTTP 500 Internal Server Error
```

The corresponding server-side logs contained a Pillow `UnidentifiedImageError`.

The exception was observed in the server logs rather than directly returned as a traceback to the client.

**Recommended hardening:**

* Validate file type
* Validate file size
* Validate image structure
* Handle decoding exceptions
* Return controlled client errors such as HTTP 400

---

## 11.5 No explicit upload-size validation

**Status:** Observed hardening weakness

A controlled 20 MB upload was accepted by the API.

The test did not demonstrate resource exhaustion or denial of service.

**Recommended hardening:**

* Maximum request size
* Maximum image size
* Content-type validation
* Reverse-proxy limits
* Resource monitoring

---

## 11.6 Container runs as root

**Status:** Demonstrated

The container returned:

```text
root
```

from:

```powershell
docker exec task-1 whoami
```

**Recommended hardening:**

Run the application using a dedicated non-root user with only the permissions required by the service.

---

## 11.7 No explicit Docker CPU or memory limits

**Status:** Demonstrated

The container had no explicit memory or CPU limits configured.

This does not by itself demonstrate a denial-of-service vulnerability.

**Recommended hardening:**

Configure appropriate:

* Memory limits
* CPU limits
* Process/PID limits
* Container restart policies
* Monitoring

---

## 11.8 Model-loading hardening

The model is loaded using PyTorch serialization.

Inspection of the legitimate model artifact using:

```python
torch.load(
    "models/mnist_cnn.pth",
    map_location="cpu",
    weights_only=True
)
```

showed an `OrderedDict` containing the expected model parameters.

**Recommended hardening:**

* Prefer `weights_only=True` where applicable.
* Validate model artifact structure.
* Maintain artifact integrity controls.
* Restrict the location and permissions of model files.
* Avoid loading untrusted serialized model objects.

This was treated as a hardening issue rather than a demonstrated exploit against the current artifact.

---

## 11.9 Confidence exposure

The API returns the model's softmax probability for the predicted class.

**Security consideration:**

Detailed confidence information can provide additional information to an attacker performing repeated queries or model probing.

**Recommended hardening:**

Evaluate whether confidence values need to be exposed externally and consider limiting returned information where appropriate.

---

# 12. Threat Model

The primary attacker is assumed to have access to the model's inference interface but not the model's internal training process.

Potential capabilities include:

* Submitting arbitrary images
* Repeating inference requests
* Observing predicted classes
* Observing returned confidence values
* Constructing adversarial inputs
* Sending malformed input files

The assessment focuses on the locally deployed system and does not target external infrastructure.

---

# 13. MITRE ATLAS Mapping

Relevant findings were mapped to MITRE ATLAS techniques.

| Technique                                 | Relevance                                                                                                      |
| ----------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| AML.T0043 – Craft Adversarial Data        | Adversarial examples were constructed against the model.                                                       |
| AML.T0015 – Evade ML Model                | Adversarial inputs successfully caused model misclassification.                                                |
| AML.T0040 – AI Model Inference API Access | The assessment interacted with the model through its exposed inference API.                                    |
| AML.T0029 – Denial of ML Service          | Relevant as a potential risk from unrestricted requests/resources, but denial of service was not demonstrated. |

The mapping distinguishes demonstrated attacks from potential techniques that were not successfully exploited.

---

# 14. NIST AI RMF Mapping

The findings were considered across the four NIST AI RMF functions:

### GOVERN

Establish security responsibilities, controls, access policies, and operational requirements.

### MAP

Identify the AI system, attack surface, threat model, and potential risks.

### MEASURE

Measure model behavior, adversarial robustness, API behavior, and security controls.

### MANAGE

Prioritize and address identified risks through controls such as authentication, validation, rate limiting, container hardening, and secure model loading.

---

# 15. Recommended Hardening

The following controls are recommended for a production deployment:

### API

* Add authentication and authorization.
* Add request rate limiting.
* Validate uploaded file types.
* Enforce maximum upload sizes.
* Handle malformed images safely.
* Avoid unnecessary information disclosure.

### Model Loading

* Use safe model-loading mechanisms.
* Prefer `weights_only=True` for compatible state-dictionary loading.
* Validate expected model structure.
* Protect model artifacts from unauthorized modification.
* Consider integrity verification.

### Docker

* Run as a non-root user.
* Configure CPU and memory limits.
* Configure process limits.
* Minimize container privileges.
* Use a minimal production image.
* Monitor resource usage.

### AI Security

* Monitor adversarial input patterns.
* Evaluate adversarial robustness periodically.
* Maintain model and artifact integrity controls.
* Monitor inference activity.
* Establish an incident-response process for model abuse.

---

# 16. Reproducing the Project

## Prerequisites

* Python 3.11
* Docker Desktop
* Git
* Windows PowerShell or equivalent shell

---

## Create the virtual environment

```powershell
py -3.11 -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution for the current session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then activate again:

```powershell
.\.venv\Scripts\Activate.ps1
```

---

## Install dependencies

```powershell
pip install -r requirements.txt
```

For the API:

```powershell
pip install -r requirements-api.txt
```

---

## Train the model

From the project root:

```powershell
python src/model/train.py
```

The trained model is saved to:

```text
models/mnist_cnn.pth
```

---

## Evaluate the model

```powershell
python src/model/evaluate.py
```

Expected baseline:

```text
98.79% accuracy
```

---

# 17. Run FGSM

Run the FGSM search:

```powershell
python src/attacks/fgsm_search.py
```

Generated examples are stored under:

```text
adversarial_examples/fgsm/
```

The visualization can be generated with:

```powershell
python src/attacks/visualize_fgsm.py
```

---

# 18. Run PGD

Run the PGD search:

```powershell
python src/attacks/pgd_search.py
```

Generated adversarial tensors are stored under:

```text
adversarial_examples/pgd/
```

Convert saved PGD tensors into image files:

```powershell
python src/attacks/save_pgd_images.py
```

---

# 19. Run the API Locally

From the project root:

```powershell
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Health check:

```powershell
Invoke-RestMethod http://localhost:8000/health
```

---

# 20. Run with Docker

Build and start the service:

```powershell
docker compose up --build
```

The API should be available at:

```text
http://localhost:8000
```

Health check:

```powershell
Invoke-RestMethod http://localhost:8000/health
```

---

# 21. API Testing

Example prediction request:

```powershell
curl.exe -X POST "http://localhost:8000/predict" `
  -F "file=@.\path\to\image.png"
```

The API returns:

```json
{
  "class": 8,
  "confidence": 0.7007153034210205
}
```

---

# 22. Security Testing

### Authentication

Submit a valid image without an authorization header:

```powershell
curl.exe -X POST "http://localhost:8000/predict" `
  -F "file=@.\path\to\image.png"
```

A successful prediction demonstrates that authentication is not required by the current endpoint.

### Rate limiting

Controlled repeated requests can be tested with:

```powershell
1..20 | ForEach-Object {
    $status = curl.exe -s -o NUL -w "%{http_code}" `
        -X POST "http://localhost:8000/predict" `
        -F "file=@.\path\to\image.png"

    "$_ : $status"
}
```

The assessment observed successful responses without an HTTP 429 response during the controlled test.

---

# 23. Docker Security Checks

Check the container user:

```powershell
docker exec task-1 whoami
```

Observed:

```text
root
```

Check configured memory limit:

```powershell
docker inspect task-1 --format "{{.HostConfig.Memory}}"
```

Check configured CPU limit:

```powershell
docker inspect task-1 --format "{{.HostConfig.NanoCpus}}"
```

A value of `0` indicates no explicit limit was configured through those Docker settings.

---

# 24. Limitations

This assessment has several important limitations:

* Testing was performed against a small MNIST model.
* Adversarial attack evaluation used the first 1,000 test images rather than the complete dataset.
* Attack success rates therefore describe the tested subset and configuration.
* The API-level validation used serialized image files, which can alter adversarial perturbations.
* Resource-exhaustion and denial-of-service attacks were not demonstrated.
* No container escape was attempted or demonstrated.
* No model extraction or membership-inference attack was performed.
* No poisoning or backdoor attack was performed.
* The system is a local assessment target rather than a production deployment.

These limitations are intentionally documented to distinguish demonstrated findings from potential risks.

---

# 25. Key Results

### Model

```text
MNIST test accuracy: 98.79%
```

### FGSM

```text
Maximum tested success rate: 86.64%
Epsilon: 0.80
```

### PGD

```text
Maximum tested success rate: 55.97%
Epsilon: 0.30
Alpha: 0.01
Steps: 40
```

### API

```text
Successful adversarial example validated
through the Dockerized inference API.
```

### Pipeline

Identified weaknesses include:

```text
✓ No authentication
✓ No observed rate limiting
✓ Invalid input → HTTP 500
✓ No explicit upload-size validation
✓ Container runs as root
✓ No explicit CPU/memory limits
✓ Model-loading hardening opportunity
✓ Confidence information exposed
```

---

# 26. Final Assessment

The assessment demonstrates that the AI system is vulnerable at both the **model layer** and the **serving-pipeline layer**.

At the ML layer, gradient-based adversarial attacks successfully caused misclassification. At the system layer, the inference API was found to lack authentication and observable rate limiting, while input validation and container hardening require additional controls.

A key finding from the end-to-end testing was that not every adversarial tensor that succeeds directly against the model remains successful after serialization and API preprocessing. This demonstrates the importance of evaluating AI security across the complete inference pipeline rather than only against the underlying model.

The recommended mitigations focus on secure API access, input validation, rate limiting, safe model loading, container least privilege, resource controls, monitoring, and adversarial robustness evaluation.

---

## 27. Assessment Evidence

The repository contains:

* Model implementation
* Training and evaluation code
* FGSM attack implementation
* PGD attack implementation
* Adversarial examples
* FastAPI serving implementation
* Docker configuration
* Attack results
* Security assessment report
* AI usage log
* Video walkthrough

All testing was performed within the authorized local assessment environment.

---

## 28. Responsible Use

This project was conducted as an authorized security assessment of a locally controlled AI system.

The techniques demonstrated here should only be applied to systems for which the tester has explicit authorization.
