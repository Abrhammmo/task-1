# AI Security Red Team Attack Report

## 1. Executive Summary

This assessment evaluates the security of a locally deployed AI image-classification system from both the machine-learning model layer and the serving pipeline.

The target consists of a PyTorch convolutional neural network trained on the MNIST handwritten-digit dataset and exposed through a FastAPI inference service running inside Docker. The assessment was conducted entirely within a controlled local environment.

The red-team assessment focused on two primary attack surfaces:

1. **Machine-learning model security**

   * Fast Gradient Sign Method (FGSM)
   * Projected Gradient Descent (PGD)
   * Adversarial examples and classification manipulation

2. **AI serving pipeline security**

   * Unauthenticated inference access
   * Input validation
   * Repeated-request/rate-limit behavior
   * Container privileges
   * Resource controls
   * Model loading and deserialization
   * Error handling

The baseline model achieved **98.79% accuracy** on the 10,000-image MNIST test set.

Adversarial testing demonstrated that the model is vulnerable to small input perturbations. FGSM successfully changed classifications, and PGD achieved higher attack success as the perturbation budget increased.

Importantly, adversarial examples were also tested against the actual inference API. This confirmed that some adversarial examples generated during the white-box attack stage survived the image conversion and API preprocessing pipeline.

The assessment also identified several serving-layer security weaknesses, including an unauthenticated prediction endpoint, no observed rate limiting during controlled repeated-request testing, insufficient explicit upload validation, execution of the container as `root`, absence of explicit Docker CPU/memory limits, and model loading without explicitly enabling safer `weights_only` deserialization.

These findings demonstrate that securing an AI application requires protection of both the machine-learning model and the surrounding application infrastructure.

---

# 2. Assessment Scope

## 2.1 Target

The assessment target was a locally developed MNIST image-classification system consisting of:

* PyTorch CNN model
* MNIST dataset
* FastAPI inference API
* Docker container
* Local HTTP endpoint

The model accepts an image and returns:

```json
{
  "class": 8,
  "confidence": 0.7007153034210205
}
```

## 2.2 Assessment Boundaries

All testing was performed against infrastructure created and controlled for this assessment.

The assessment did not target:

* External websites
* Third-party APIs
* Production systems
* Other people's infrastructure
* Public AI services

The attacks were performed locally against the assessment model and its Dockerized API.

---

# 3. Target Architecture

The target architecture can be represented as:

```text
                 Red Team Attacker
                        |
                        |
                 HTTP POST /predict
                        |
                        v
              +--------------------+
              |     FastAPI API    |
              |   Port 8000        |
              +---------+----------+
                        |
                        v
              +--------------------+
              | Image Preprocessing|
              | Grayscale          |
              | Resize 28x28       |
              | Normalize          |
              +---------+----------+
                        |
                        v
              +--------------------+
              |    MNIST CNN       |
              |   PyTorch Model    |
              +---------+----------+
                        |
                        v
              Class + Confidence
```

The API and model are packaged inside a Docker container.

The red-team assessment therefore considered both the model itself and the components surrounding it.

---

# 4. Threat Model

The assumed attacker has access to the inference endpoint and can submit images to the model.

The attacker may attempt to:

* Cause incorrect model predictions
* Manipulate model confidence
* Discover weaknesses in the inference interface
* Submit malformed files
* Send repeated inference requests
* Abuse unrestricted resource consumption
* Take advantage of insecure container configuration
* Investigate model-loading weaknesses

The assessment primarily represents a **black-box/gray-box application attacker** at the API layer and a **white-box attacker** at the model layer.

---

# 5. Baseline Model Assessment

## 5.1 Model Architecture

The target is a convolutional neural network implemented in PyTorch.

The feature extractor consists of:

* Convolutional layer with 32 channels
* ReLU activation
* Max pooling
* Convolutional layer with 64 channels
* ReLU activation
* Max pooling

The classifier consists of:

* Flatten layer
* Fully connected layer with 128 neurons
* ReLU activation
* Final fully connected layer with 10 outputs

The ten output classes correspond to the digits:

```text
0 1 2 3 4 5 6 7 8 9
```

## 5.2 Training Configuration

The model was trained using:

| Parameter     |         Value |
| ------------- | ------------: |
| Dataset       |         MNIST |
| Epochs        |             5 |
| Batch size    |            64 |
| Learning rate |         0.001 |
| Optimizer     |          Adam |
| Loss          | Cross Entropy |
| Device        |           CPU |

The model was saved as:

```text
models/mnist_cnn.pth
```

## 5.3 Baseline Accuracy

The trained model was evaluated against the complete MNIST test set.

```text
Correct predictions: 9879 / 10000
Accuracy: 98.79%
```

This provides the baseline against which adversarial degradation was measured.

---

# 6. Adversarial Attack Assessment

## 6.1 Attack Objective

The objective of the adversarial testing was to determine whether small, deliberately constructed changes to input images could cause the model to produce incorrect predictions.

Two white-box attacks were implemented:

* FGSM
* PGD

Only images that were initially classified correctly were counted when calculating attack success.

This prevents naturally misclassified images from being incorrectly counted as successful adversarial attacks.

---

# 7. FGSM Attack

## 7.1 Method

The Fast Gradient Sign Method generates an adversarial image by modifying the input in the direction of the sign of the loss gradient.

The implemented attack is:

```python
def fgsm_attack(image, epsilon, gradient):
    gradient_sign = gradient.sign()
    adversarial_image = image + epsilon * gradient_sign
    return adversarial_image
```

The perturbation budget is controlled by `epsilon`.

Higher epsilon values permit larger changes to the original image.

---

## 7.2 FGSM Results

The attack was evaluated against the first 1,000 MNIST test images.

Of these:

```text
Initially correct: 988
```

The following results were obtained:

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

The results show a clear relationship between perturbation magnitude and attack success.

As epsilon increased, a larger proportion of initially correct images were successfully misclassified.

---

## 7.3 FGSM Successful Example

One successful example at:

```text
epsilon = 0.10
```

had the following results:

| Measurement     |  Clean | Adversarial |
| --------------- | -----: | ----------: |
| True label      |      5 |           5 |
| Predicted class |      5 |           8 |
| Confidence      | 52.79% |      91.49% |

The model therefore changed from correctly identifying the digit as `5` to confidently predicting `8`.

The comparison image was saved as:

```text
adversarial_examples/fgsm/epsilon_0.10/fgsm_comparison.png
```

This provides visual and numerical evidence of a successful adversarial attack.

---

# 8. PGD Attack

## 8.1 Method

Projected Gradient Descent was implemented as an iterative extension of the gradient-based attack.

The attack repeatedly updates the adversarial image and projects the resulting perturbation back into the allowed epsilon region.

The implementation used:

```text
Alpha: 0.01
Steps: 40
```

The projection step ensures that the final perturbation remains within the selected epsilon budget.

---

## 8.2 PGD Results

The attack was evaluated against the first 1,000 test images.

Again, 988 images were initially classified correctly.

| Epsilon | Successful Attacks | Success Rate |
| ------: | -----------------: | -----------: |
|    0.05 |                 23 |        2.33% |
|    0.10 |                 48 |        4.86% |
|    0.15 |                120 |       12.15% |
|    0.20 |                247 |       25.00% |
|    0.25 |                393 |       39.78% |
|    0.30 |                553 |       55.97% |

The results show that increasing the perturbation budget substantially increases the probability of successful misclassification.

---

## 8.3 Successful PGD Examples

The first successful example found at each epsilon was recorded.

| Epsilon | Index | True | Clean | Adversarial |
| ------: | ----: | ---: | ----: | ----------: |
|    0.05 |     8 |    5 |     5 |           8 |
|    0.10 |     6 |    4 |     4 |           8 |
|    0.15 |     6 |    4 |     4 |           8 |
|    0.20 |     4 |    4 |     4 |           9 |
|    0.25 |     1 |    2 |     2 |           6 |
|    0.30 |     1 |    2 |     2 |           6 |

The strongest example at epsilon `0.05` was particularly useful for pipeline testing because the perturbation was relatively small.

---

# 9. API-Level Adversarial Testing

A key part of the assessment was determining whether adversarial examples generated against the model would remain effective after passing through the actual serving pipeline.

This distinction is important because an adversarial tensor that fools the model directly does not necessarily fool the deployed API.

The generated PGD tensors were converted into PNG images and submitted through the `/predict` endpoint.

---

## 9.1 API Results

Six saved PGD examples were tested through the API.

| Epsilon | Clean Prediction | Adversarial Prediction | API Attack Successful? |
| ------: | ---------------: | ---------------------: | ---------------------- |
|    0.05 |                5 |                      8 | Yes                    |
|    0.10 |                4 |                      4 | No                     |
|    0.15 |                4 |                      8 | Yes                    |
|    0.20 |                4 |                      4 | No                     |
|    0.25 |                2 |                      2 | No                     |
|    0.30 |                2 |                      2 | No                     |

Therefore:

```text
Successful through API: 2 / 6
```

This result demonstrates why attacks must be validated against the actual serving pipeline rather than only against the model in memory.

---

## 9.2 Demonstrated API Attack

The epsilon `0.05` adversarial example successfully passed through the Dockerized API.

The request returned:

```text
HTTP 200
```

with:

```json
{
  "class": 8,
  "confidence": 0.7007153034210205
}
```

The clean image was classified as digit `5`, while the adversarial image was classified as digit `8`.

This is a demonstrated end-to-end adversarial attack against the deployed inference pipeline.

---

# 10. API Security Assessment

## 10.1 Unauthenticated Inference Endpoint

### Observation

The `/predict` endpoint accepts inference requests without authentication.

No:

* API key
* authentication token
* session
* user credentials

were required during testing.

### Demonstration

An image was submitted directly to:

```text
POST /predict
```

and the API returned a normal prediction.

### Security Impact

If this API were exposed beyond the controlled local environment, an unauthorized party could potentially submit arbitrary inference requests.

This also makes other attacks, including adversarial-input attacks and request flooding, easier to perform.

### Finding

**Unauthenticated inference endpoint**

**Severity:** Medium in the current local demonstration; potentially higher depending on deployment exposure and business context.

---

# 11. Rate-Limiting Assessment

Repeated inference requests were sent to the `/predict` endpoint without authentication.

A controlled test sent:

```text
20 consecutive requests
```

The requests returned successful HTTP responses and no HTTP `429 Too Many Requests` response was observed.

### Result

```text
Requests sent: 20
Observed rate-limit response: None
```

### Interpretation

No rate limiting was observed during this controlled test.

This does not by itself prove that unlimited requests could be sustained indefinitely, but it indicates that the API did not visibly enforce a request limit during the test.

### Potential Impact

Without rate controls, an exposed inference service could be more susceptible to:

* Excessive inference usage
* Resource consumption
* Automated probing
* Model behavior probing
* Denial-of-service attempts

---

# 12. Invalid Input Handling

An invalid `.pt` file was submitted to the `/predict` endpoint.

The API returned:

```text
HTTP 500 Internal Server Error
```

The Docker logs showed a:

```text
PIL.UnidentifiedImageError
```

generated when Pillow attempted to process the invalid file.

### Security Impact

The endpoint does not currently handle invalid image input gracefully.

A malformed or unsupported file can therefore trigger an internal server exception.

The exception was visible in the server-side Docker logs rather than being returned as a detailed traceback to the client.

### Finding

**Insufficient invalid-input handling**

Recommended improvements include:

* Explicit file-type validation
* Image decoding validation
* Controlled exception handling
* Appropriate HTTP 4xx responses
* Upload size restrictions

---

# 13. File Upload and Resource Controls

The API reads the complete uploaded file into memory:

```python
image_bytes = await file.read()
```

No explicit application-level upload size restriction was identified.

A controlled 20 MB arbitrary binary file was submitted successfully enough to reach image processing and resulted in a prediction response rather than an upload-size rejection.

The test demonstrated that large input was accepted, but it did **not** demonstrate resource exhaustion or denial of service.

### Finding

**Missing explicit upload-size validation**

### Potential Impact

If the service were externally exposed, unrestricted uploads could increase memory and processing consumption.

Recommended controls include:

* Maximum request body size
* Maximum image dimensions
* Content-type validation
* File signature validation
* Request timeouts
* Rate limiting

---

# 14. Docker Container Privileges

The running container was inspected using:

```powershell
docker exec task-1 whoami
```

The result was:

```text
root
```

The container therefore runs processes as the `root` user.

### Security Impact

Running the application as root increases the potential impact of a successful application-level compromise.

A vulnerability inside the API or its dependencies could provide an attacker with root privileges inside the container.

This test did not demonstrate container escape.

### Finding

**Container runs as root**

Recommended hardening:

* Create a dedicated non-root user
* Run the application using that user
* Minimize Linux capabilities
* Consider a read-only filesystem where practical
* Apply container security profiles

---

# 15. Docker Resource Limits

The container configuration was inspected for memory and CPU limits.

The observed configuration indicated:

```text
Memory limit: 0
CPU limit: 0
```

In this Docker configuration, these values indicate that no explicit container-level memory or CPU limit was configured.

### Security Impact

An externally reachable inference service without resource controls may be more exposed to resource exhaustion.

Potentially expensive requests or large numbers of requests could consume host resources.

### Important Limitation

A denial-of-service attack was **not demonstrated** during this assessment.

The finding is therefore treated as a configuration/hardening weakness rather than a successfully exploited resource-exhaustion vulnerability.

### Recommended Controls

* Set memory limits
* Set CPU limits
* Configure process limits
* Apply request-rate limits
* Monitor container resource usage

---

# 16. Model Loading Assessment

The API loads the model using:

```python
torch.load(
    MODEL_PATH,
    map_location=device
)
```

The model artifact was separately inspected using:

```python
torch.load(
    "models/mnist_cnn.pth",
    map_location="cpu",
    weights_only=True
)
```

The legitimate model artifact was confirmed to be an `OrderedDict` containing eight expected model parameter tensors.

Observed keys included:

```text
features.0.weight
features.0.bias
features.3.weight
features.3.bias
classifier.1.weight
classifier.1.bias
classifier.3.weight
classifier.3.bias
```

### Finding

The API does not explicitly specify:

```python
weights_only=True
```

when loading the model.

### Security Consideration

For a trusted local artifact, this did not produce an observed exploit during the assessment.

However, explicit safer deserialization settings and artifact integrity validation should be used when loading model files, particularly when artifacts could originate from an untrusted source.

### Recommended Hardening

Use:

```python
torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True
)
```

and validate the expected model structure before loading it into the service.

---

# 17. Confidence Exposure

The API returns both the predicted class and its confidence:

```json
{
  "class": 8,
  "confidence": 0.7007153034210205
}
```

Confidence information can be useful to legitimate consumers, but it can also provide additional information to an attacker performing repeated model queries.

### Potential Security Impact

An attacker may use prediction probabilities to understand model behavior and potentially improve automated probing or adversarial-input generation.

No model extraction or membership-inference attack was performed during this assessment.

Therefore, confidence exposure is treated as a potential information-leak/hardening concern rather than a demonstrated extraction vulnerability.

---

# 18. Findings Summary

| ID   | Finding                                          | Evidence                            | Status              |
| ---- | ------------------------------------------------ | ----------------------------------- | ------------------- |
| F-01 | Adversarial examples can manipulate predictions  | FGSM and PGD testing                | **Demonstrated**    |
| F-02 | Adversarial example survives API pipeline        | PGD epsilon 0.05 through Docker API | **Demonstrated**    |
| F-03 | Inference endpoint has no authentication         | Direct `/predict` request           | **Demonstrated**    |
| F-04 | No rate limiting observed                        | 20 controlled requests              | **Observed**        |
| F-05 | Invalid files cause HTTP 500                     | Invalid `.pt` upload                | **Demonstrated**    |
| F-06 | Explicit upload-size validation absent           | 20 MB upload accepted               | **Observed**        |
| F-07 | Container runs as root                           | `docker exec task-1 whoami`         | **Demonstrated**    |
| F-08 | No explicit CPU/memory limits                    | Docker inspection                   | **Observed**        |
| F-09 | Model loading lacks explicit `weights_only=True` | Source inspection                   | **Hardening issue** |
| F-10 | Confidence is exposed                            | `/predict` response                 | **Observed**        |

---

# 19. Hardening Recommendations

## 19.1 Protect the Inference Endpoint

Implement authentication and authorization before allowing inference requests.

Possible controls include:

* API keys
* OAuth/JWT authentication
* Network-level access controls
* Role-based authorization

---

## 19.2 Implement Rate Limiting

Apply request limits per:

* IP address
* API key
* authenticated user
* client identity

Also consider request quotas and burst limits.

---

## 19.3 Validate Uploaded Files

The API should validate:

* Content type
* File extension
* File signature
* Maximum file size
* Maximum image dimensions
* Successful image decoding

Invalid files should receive controlled `4xx` responses rather than causing unhandled exceptions.

---

## 19.4 Apply Resource Limits

Configure Docker CPU and memory limits.

Example configuration:

```yaml
services:
  api:
    mem_limit: 1g
    cpus: "1.0"
```

The exact values should be selected based on the expected workload rather than treated as universal security settings.

---

## 19.5 Run the Container as a Non-Root User

Create a dedicated application user inside the Docker image and execute the API under that account.

This reduces the impact of application-level compromise.

---

## 19.6 Harden Model Loading

Use safer model loading:

```python
torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True
)
```

Also:

* Restrict model artifact permissions
* Validate expected model structure
* Maintain trusted artifact sources
* Consider integrity hashes
* Prevent unauthorized replacement of model files

---

## 19.7 Consider Adversarial Robustness Controls

Potential defenses against adversarial examples include:

* Adversarial training
* Input preprocessing
* Robust model architectures
* Input anomaly detection
* Ensemble approaches
* Confidence calibration
* Adversarial monitoring

No single defense should be assumed to eliminate adversarial examples completely.

---

## 19.8 Reduce Unnecessary Information Exposure

Depending on application requirements, consider whether exact confidence values need to be returned to clients.

If confidence is not required, returning only the predicted class can reduce information available for model probing.

---

# 20. MITRE ATLAS Mapping

The assessment findings can be considered in the context of MITRE ATLAS, which provides a framework for describing threats and techniques targeting AI-enabled systems.

Relevant attack themes demonstrated or investigated during the assessment include:

| Assessment Activity           | ATLAS-Relevant Theme                       |
| ----------------------------- | ------------------------------------------ |
| FGSM adversarial examples     | Adversarial ML / evasion                   |
| PGD adversarial examples      | Adversarial ML / evasion                   |
| API probing                   | AI system reconnaissance                   |
| Repeated inference requests   | Model/API probing                          |
| Confidence observation        | Information gathering about model behavior |
| Model artifact loading review | AI artifact/model security                 |
| Docker security review        | Supporting infrastructure security         |

The most direct demonstrated AI attack in this assessment was adversarial evasion, where carefully modified inputs caused the classifier to produce incorrect predictions.

---

# 21. NIST AI RMF Mapping

The findings can also be considered using the four core functions of the NIST AI Risk Management Framework:

## Govern

Security requirements should be defined for the AI model and its serving infrastructure.

Relevant recommendations include:

* Define model-security responsibilities
* Establish artifact-management controls
* Define authentication requirements
* Establish security testing procedures

## Map

The assessment identified risks across multiple layers:

```text
Model
  |
  +-- Adversarial examples
  |
API
  |
  +-- Authentication
  +-- Input validation
  +-- Rate limiting
  |
Container
  |
  +-- Root privileges
  +-- Resource limits
  |
Model Artifact
  |
  +-- Loading/deserialization
```

## Measure

The assessment measured:

* Baseline accuracy
* FGSM success rate
* PGD success rate
* API-level adversarial success
* API responses to invalid inputs
* Authentication behavior
* Rate-limit behavior
* Container privilege configuration
* Resource-limit configuration

## Manage

Recommended risk treatments include:

* Authentication
* Rate limiting
* Input validation
* Resource limits
* Non-root containers
* Safer model loading
* Adversarial robustness testing
* Security monitoring

---

# 22. Reproducibility

The assessment can be reproduced using the project repository.

Important components include:

```text
src/
├── model/
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── attacks/
│   ├── fgsm.py
│   ├── fgsm_search.py
│   ├── pgd.py
│   ├── pgd_search.py
│   ├── save_pgd_images.py
│   └── visualize_fgsm.py
│
└── api/
    └── main.py
```

Generated adversarial examples are stored under:

```text
adversarial_examples/
├── fgsm/
└── pgd/
```

The model is stored at:

```text
models/mnist_cnn.pth
```

The serving environment is defined through:

```text
Dockerfile
docker-compose.yml
requirements-api.txt
```

---

# 23. Limitations

Several limitations should be considered when interpreting the results.

### Dataset Scope

The model uses MNIST, which is a relatively simple image-classification problem. Results should not automatically be generalized to more complex production models.

### Attack Scope

The assessment focused primarily on white-box FGSM and PGD attacks.

Model extraction, membership inference, poisoning, and backdoor attacks were not performed.

### API Rate Testing

Only controlled repeated-request testing was performed. The absence of an observed HTTP `429` response does not prove that the service has no limits under every possible workload.

### Resource Exhaustion

A 20 MB upload was accepted, but resource exhaustion was not demonstrated.

### Container Escape

The container running as root was demonstrated, but no container escape was attempted or demonstrated.

### Model Loading

The model-loading configuration was reviewed for hardening purposes. No malicious model artifact was used to exploit the loading mechanism.

### API Adversarial Success

Not every white-box PGD example remained successful after conversion to PNG and API preprocessing. This demonstrates that model-level attack success and end-to-end API attack success are different measurements.

---

# 24. Overall Assessment

The assessment demonstrated that the AI system has security weaknesses at both the model and serving layers.

At the model layer, FGSM and PGD successfully generated adversarial inputs capable of changing predictions. Increasing the perturbation budget increased attack success rates.

At the serving layer, an adversarial example generated at the model level was successfully transferred through the image-processing and Dockerized FastAPI pipeline and caused an incorrect prediction.

The API assessment additionally identified unauthenticated inference access, no observed rate limiting during controlled testing, insufficient explicit upload validation, unhandled invalid-image exceptions, root container execution, and missing explicit resource limits.

The assessment therefore shows that AI security cannot be treated solely as a model-accuracy problem. The model, API, container, model artifact, and surrounding controls form a single attack surface.

The most important security lesson from this assessment is that an adversarial attack should be validated end-to-end. A perturbation that fools the model in memory is not necessarily effective after serialization, image conversion, preprocessing, and API inference. Testing the complete pipeline provided stronger evidence of practical attack impact.

---

# 25. Recommended Priority Actions

The following actions should be prioritized for a production deployment:

1. **Require authentication and authorization for inference requests.**
2. **Implement request rate limiting and quotas.**
3. **Add strict image type, size, and dimension validation.**
4. **Handle invalid images with controlled `4xx` responses.**
5. **Run the Docker container as a non-root user.**
6. **Apply CPU and memory limits to the container.**
7. **Use safer model loading with `weights_only=True`.**
8. **Validate and protect model artifacts.**
9. **Introduce adversarial robustness testing into the ML lifecycle.**
10. **Monitor model and API behavior for abnormal probing and attack activity.**

---

# 26. Conclusion

This red-team assessment evaluated the security of a complete local AI inference system rather than examining model accuracy alone.

The assessment demonstrated successful adversarial manipulation using FGSM and PGD and validated that selected adversarial examples could survive the actual API and Docker serving pipeline.

The assessment also identified weaknesses in authentication, request controls, input validation, container privileges, resource configuration, and model-loading hardening.

The results provide a practical security baseline for the system and identify controls that should be implemented before exposing a similar AI inference service to an untrusted network.
