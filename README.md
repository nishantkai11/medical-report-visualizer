# Multimodal Medical Report Visualizer

A single-file Streamlit research prototype combining pretrained chest-X-ray computer vision, biomedical NLP, report parsing, image-report comparison, plain-language explanation, visualization, and JSON export.

## Deploy

This repository is intentionally flattened so it can be uploaded directly through the GitHub website without Git, GitHub Desktop, or a terminal.

Use Streamlit Community Cloud and set the main file to `app.py`.

## Responsible AI

This is an educational/research prototype, not a diagnostic system. Do not upload identifiable patient information. AI outputs are model predictions/observations, not diagnoses. The written radiology/clinician report remains the primary clinical source. Agreement with a report is not proof of model accuracy. No treatment recommendations are generated.

## Models

- `itsomk/chexpert-densenet121`
- `d4data/biomedical-ner-all`
