# iot-anomaly-detection-xai

Human-Centred IoT Sensor Anomaly Detection with XAI and Natural Language Explanations.
---

## Overview

This project builds a human-in-the-loop (HITL) system for IoT sensor anomaly detection. When an anomaly is flagged, a local LLM generates a plain-language explanation grounded in temporal and domain context. A Streamlit dashboard lets operators confirm, reject, or snooze alerts — closing the feedback loop.
---

## Research Questions

**RQ1 (primary):** Does providing LLM-generated natural language explanations in an IoT anomaly detection system increase user trust and improve anomaly assessment compared to a system without explanations?

**RQ2:** How do different prompting strategies (zero-shot, contextualised, RAG) affect the quality of generated explanations, measured via LLM-as-Judge?

**RQ3:** How do users perceive the usability and trustworthiness of the HCAI system compared to a baseline?

---

## System Architecture

1. Data ingestion/preprocessing
2. Anomaly Detection
3. Context Assembly (Local and Global context)
4. LLM Explainer 
5. Streamlit UI

---

## Dataset

AI4I 2020 Predictive Maintenance | 10,000 | Air temp, process temp, speed, torque, tool wear | 5 failure modes (HDF, PWF, OSF, TWF, RNF) | 
Source: AI4I 2020 Predictive Maintenance Dataset [Dataset]. (2020). UCI Machine Learning Repository. https://doi.org/10.24432/C5HS5C.

---

## Anomaly detection

WIP

---

## Anomaly Context Payload

Temporal context (+-10 readings), rolling 24h statistics (median, std), and global percentile bands (p5–p95) replace feature attribution scores, making explanations interpretable without ML expertise.

WIP

---

## RAG Knowledge Base

WIP
---

## Evaluation

WIP
---

## References

- Chandola, V., Banerjee, A., & Kumar, V. (2009). Anomaly detection: A survey. *ACM Computing Surveys*, 41(3).
- Chatterjee, A., & Ahmed, B. S. (2022). IoT anomaly detection methods and applications: A survey. *Internet of Things*, 19.
- Li, Z., Zhu, Y., & van Leeuwen, M. (2023). A survey on explainable anomaly detection. *ACM TKDD*, 18(1).
- Gummadi, N., et al. (2024). XAI-IoT: An explainable AI framework for enhancing anomaly detection in IoT systems. *IEEE Access*.
- Padín-Torrente, H., et al. (2026). Toward human-centered explainability: Natural language explanations for anomaly detection. *Information Systems Frontiers*.
