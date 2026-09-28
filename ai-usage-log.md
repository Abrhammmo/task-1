# AI Usage Log

## 1. AI Tool Information

**AI Tool:** ChatGPT
**Model:** GPT-5.6 Luna
**Provider:** OpenAI
**Primary purpose:** Technical assistance, security assessment guidance, code explanation, troubleshooting, attack methodology development, analysis of results, report drafting, and MITRE ATLAS / NIST AI RMF mapping.

AI was used as an assistant throughout the assessment. All code, commands, security tests, experimental results, and conclusions were reviewed and executed by me. AI-generated suggestions were adapted where necessary to match the actual project implementation and observed results.

---

## 2. AI Usage Record

| #  | Purpose                                   | Prompt / Prompt Summary                                                                                                                                                                   | How the AI Output Was Used or Changed                                                                                                                                                                                                                                                                                      |
| -- | ----------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1  | Understanding the assessment requirements | Asked ChatGPT to analyze and explain the AI Security Red Team Assessment requirements and expected deliverables.                                                                          | Used to structure the work into target construction, baseline assessment, adversarial attacks, API/pipeline testing, documentation, video, and AI usage logging. The assessment brief remained the authoritative source.                                                                                                   |
| 2  | Project and environment setup             | Asked for guidance on Python versions, virtual environments, dependencies, PyTorch, FastAPI, and Docker setup.                                                                            | Used the commands and explanations to configure the local development environment. Commands were executed manually and errors were independently verified and corrected.                                                                                                                                                   |
| 3  | Model development                         | Asked for assistance understanding and implementing a simple MNIST CNN using PyTorch.                                                                                                     | Used the explanation and code structure to build the local model. The resulting model was trained locally and its 98.79% test accuracy was independently measured.                                                                                                                                                         |
| 4  | Baseline evaluation                       | Asked how to evaluate the trained model and interpret classification accuracy/confidence.                                                                                                 | Used the guidance to implement/run the baseline evaluation. The reported accuracy and predictions were taken from actual local execution rather than assumed AI results.                                                                                                                                                   |
| 5  | FGSM attack                               | Asked for an explanation and implementation of the Fast Gradient Sign Method (FGSM) against the local MNIST model.                                                                        | Used the methodology to implement the FGSM attack locally. Epsilon values and attack results were tested experimentally.                                                                                                                                                                                                   |
| 6  | FGSM attack analysis                      | Asked ChatGPT to help interpret FGSM results and success rates.                                                                                                                           | Used the analysis to organize the results and explain the relationship between increasing epsilon and attack success. Numerical results in the report came from my own executions.                                                                                                                                         |
| 7  | PGD attack                                | Asked for assistance implementing and understanding Projected Gradient Descent (PGD) as a stronger iterative white-box attack.                                                            | Used the methodology to implement the local PGD attack. The attack parameters, including epsilon, alpha, and number of steps, were selected and tested on the local model.                                                                                                                                                 |
| 8  | PGD result analysis                       | Asked ChatGPT to interpret PGD attack results and distinguish successful misclassification from confidence reduction.                                                                     | Used this distinction in the report. In particular, API-level results were checked separately instead of assuming that every white-box PGD example remained successful after image conversion and API preprocessing.                                                                                                       |
| 9  | API implementation                        | Asked for assistance building a minimal FastAPI inference endpoint for the MNIST model.                                                                                                   | Used the guidance to implement `/health` and `/predict`. The API was run locally and tested manually.                                                                                                                                                                                                                      |
| 10 | Docker deployment                         | Asked for help containerizing the FastAPI model-serving application and troubleshooting Docker build/runtime issues.                                                                      | Used the guidance to build and run the Docker container locally. Container behavior was verified with Docker commands.                                                                                                                                                                                                     |
| 11 | Adversarial API testing                   | Asked how to test adversarial images against the actual serving API rather than only against the Python model.                                                                            | Used the approach to submit generated adversarial PNG files to `/predict`. Results were independently recorded from HTTP responses.                                                                                                                                                                                        |
| 12 | Docker/API security assessment            | Asked for guidance on identifying security weaknesses in the model-serving pipeline, including file validation, rate limiting, container privileges, resource limits, and error handling. | Used the suggestions to design controlled local tests. Only weaknesses supported by observed evidence were classified as demonstrated findings.                                                                                                                                                                            |
| 13 | Invalid file testing                      | Asked how to test API behavior with invalid/non-image files.                                                                                                                              | Uploaded an invalid file to the local API and observed the HTTP 500 response and corresponding server-side Pillow exception in Docker logs. The report explicitly distinguishes this from client-side traceback disclosure.                                                                                                |
| 14 | Upload/resource testing                   | Asked about testing file-size handling and possible resource-exhaustion risks.                                                                                                            | Conducted a controlled 20 MB upload test. The file was accepted, but resource exhaustion was not demonstrated. The report therefore records missing explicit size validation as a hardening concern rather than claiming a successful DoS attack.                                                                          |
| 15 | Container security                        | Asked how to inspect Docker container privileges and resource limits.                                                                                                                     | Used Docker inspection commands to verify that the container ran as `root` and had no explicit CPU or memory limits configured. No container escape or denial-of-service was claimed.                                                                                                                                      |
| 16 | Model loading security                    | Asked about secure PyTorch model loading and `weights_only`.                                                                                                                              | Used the recommendation to inspect the legitimate model artifact. `torch.load(..., weights_only=True)` confirmed that the artifact was an `OrderedDict` containing the expected model state-dict tensors. The report treats explicit `weights_only=True` as a hardening recommendation rather than a demonstrated exploit. |
| 17 | Confidence exposure                       | Asked about the security implications of returning model confidence through the API.                                                                                                      | Included confidence exposure as an information-disclosure/probing concern. No model extraction attack was claimed because model extraction was not demonstrated.                                                                                                                                                           |
| 18 | MITRE ATLAS                               | Prompt: **“what are MITRE ATLAS and NIST AI RMF”**                                                                                                                                        | Used the explanation to understand the security frameworks and later map demonstrated attack techniques to relevant ATLAS techniques.                                                                                                                                                                                      |
| 19 | MITRE ATLAS / NIST AI RMF mapping         | Asked ChatGPT to help map the observed attacks and security findings to MITRE ATLAS and NIST AI RMF.                                                                                      | Used the mapping as a documentation aid. The final report distinguishes demonstrated techniques from potential risks and hardening recommendations.                                                                                                                                                                        |
| 20 | Security assessment documentation         | Asked ChatGPT to continue with the security assessment, hardening suggestions, remediation recommendations, and MITRE ATLAS / NIST AI RMF alignment.                                      | Used AI-generated structure and wording as a starting point. The report was adapted to reflect the actual architecture, commands, attack results, and evidence obtained during testing.                                                                                                                                    |
| 21 | Report review                             | Asked ChatGPT to review whether the assessment documentation was complete and whether remediation was actually required by the brief.                                                     | Used the review to distinguish required attack/report deliverables from optional or useful hardening implementation. The assessment focuses on demonstrated vulnerabilities and documented recommendations rather than claiming that all fixes were implemented.                                                           |
| 22 | Final AI usage documentation              | Asked ChatGPT to prepare the AI usage log for submission.                                                                                                                                 | This table documents the major AI-assisted activities and how the generated material was actually used, tested, modified, or independently verified.                                                                                                                                                                       |

---

## 3. Important AI-Use Boundaries

AI assistance was used as a technical assistant and not as an autonomous agent.

The following principles were followed:

* The assessment target was built and operated locally.
* AI did not access or attack external systems.
* Attack commands and code were executed and verified by me.
* Experimental results were obtained from my own local environment.
* AI-generated code was reviewed and adapted before use.
* Security findings were not accepted solely because AI suggested them.
* Findings were classified according to available evidence.
* Where a risk was not experimentally demonstrated, it was documented as a potential risk or hardening recommendation rather than as a confirmed vulnerability.
* Adversarial examples were tested both against the model directly and, where applicable, against the actual API pipeline.
* The final report was adapted to match the actual implementation and observed results.

---

## 4. Examples of AI Output That Required Verification or Modification

Several AI suggestions required practical verification because the final assessment had to reflect the actual local environment.

### Adversarial attacks

AI provided implementation guidance for FGSM and PGD. The resulting attacks were executed locally, and the measured success rates were used instead of any values suggested by AI.

For example, PGD attacks that succeeded against the model directly were subsequently tested through the API. Only the examples that remained successful after API preprocessing were treated as API-level successful attacks.

### Docker security

AI suggested checking container privileges and resource limits. These were verified using Docker commands. The actual findings were based on observed values:

* Container user: `root`
* Memory limit: not explicitly configured
* CPU limit: not explicitly configured

No container escape or resource-exhaustion attack was claimed because neither was demonstrated.

### Invalid file handling

AI suggested testing malformed uploads. The local API produced an HTTP 500 response for an invalid file, while the corresponding Pillow exception appeared in the server logs.

The final report therefore describes this as an input-validation/error-handling weakness and does **not** claim that the server traceback was exposed directly to the client.

### Model loading

AI suggested using explicit `weights_only=True` when loading the PyTorch model. This was independently tested against the actual `mnist_cnn.pth` file.

The result was an `OrderedDict` containing the expected eight model parameter tensors. This was used as evidence about the current artifact, while explicit safe loading was documented as a hardening recommendation.

---

## 5. AI Contribution to Final Deliverables

AI contributed primarily to:

1. Technical explanations and methodology.
2. Attack implementation guidance.
3. Troubleshooting and debugging assistance.
4. Interpretation and organization of experimental results.
5. Security finding structure.
6. MITRE ATLAS and NIST AI RMF mapping assistance.
7. Report organization and wording.
8. Preparation of this AI usage log.

The following were independently performed or verified by me:

* Local environment setup.
* Model training.
* Model evaluation.
* FGSM execution.
* PGD execution.
* Adversarial image generation.
* API deployment.
* Docker deployment.
* API-level adversarial testing.
* Docker security inspection.
* Invalid-input testing.
* Model artifact inspection.
* Collection of experimental results.
* Final selection and classification of security findings.

The final assessment does not treat AI-generated suggestions as experimental evidence. Experimental evidence comes from the local system and test results.

