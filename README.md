# Service Experience Intelligence System

## Overview

The **Service Experience Intelligence System** is a Streamlit-based analytics application for understanding service quality, satisfaction patterns, feedback themes, and improvement priorities.

The core question is:

> Which aspects of the service experience are improving, deteriorating, or creating risk?

This project focuses on experience intelligence rather than operational forecasting or generic complaint monitoring.

---

## Why This Project Exists

Service feedback often contains both structured and unstructured signals: ratings, service areas, response times, comments, and satisfaction indicators. These signals can help identify where service quality is strong and where improvement may be needed.

This app helps analyze:

- satisfaction and experience scores,
- high-risk service areas,
- feedback themes,
- experience trends,
- and improvement actions.

---

## Current Capabilities

### Feedback Upload and Demo Data

The app supports:

- CSV upload for service feedback datasets,
- synthetic demo feedback data,
- time-window filtering,
- service-area filtering.

### Experience Score

The dashboard creates an experience score using rating, text sentiment signals, and response-time pressure.

This gives a simple, interpretable service-quality indicator.

### Risk Band Mix

The app visualizes the distribution of healthy, moderate, and high-risk service experiences.

### Service Area Comparison

Service areas are compared by feedback volume and average experience score.

### Text Theme Signals

The app uses TF-IDF to extract important terms from open-text feedback.

This provides a lightweight way to surface recurring experience themes without requiring heavy NLP infrastructure.

### Trend and Deterioration View

The app tracks monthly experience-score changes to help identify deterioration or improvement over time.

### Improvement Actions

The recommendation section converts service experience patterns into practical improvement suggestions.

---

## Design Choice

This project intentionally uses lightweight, interpretable NLP rather than a complex black-box language model.

The goal is to create a practical service-improvement dashboard that is easy to run, easy to explain, and suitable for portfolio demonstration.

---

## Technology Stack

- Python
- Streamlit
- Pandas
- Plotly
- Scikit-Learn
- TF-IDF Vectorization

---

## Example Use Cases

- Service feedback analysis
- Learner support experience review
- Internal service quality monitoring
- Customer experience monitoring
- Support-channel improvement planning
- Satisfaction trend analysis

---

## Recommended Dataset Columns

The app works best with fields similar to:

- feedback date
- service area
- rating
- feedback text
- response time
- channel
- issue category

Synthetic demo data is included so the app can run without external data.

---

## How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Future Enhancements

Possible improvements include:

- VADER or transformer-based sentiment scoring,
- topic modeling using NMF or LDA,
- service deterioration alerts,
- satisfaction driver modeling,
- response-time impact analysis,
- experience segmentation,
- dashboard export features.

---

## Project fit

This project fits the **Decision Support / Service Experience Analytics** area.

It shows how feedback can be converted into clearer service-improvement priorities without requiring a large production system.

The core decision-support question is:

> What should be improved first, and why?
