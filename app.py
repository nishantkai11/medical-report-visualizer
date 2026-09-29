# Multimodal Medical Report Visualizer
# Single-file deployment build for Streamlit Community Cloud.

import os
from typing import Any

import numpy as np
from PIL import Image

LABELS = [
    "atelectasis", "cardiomegaly", "consolidation", "edema",
    "effusion", "emphysema", "fibrosis", "hernia",
    "infiltration", "mass", "nodule", "pleural_thickening",
    "pneumonia", "pneumothorax"
]

_PIPELINE = None

def _load_pipeline():
    global _PIPELINE
    if _PIPELINE is not None:
        return _PIPELINE
    try:
        from transformers import pipeline
        model = os.getenv("VISION_MODEL", "itsomk/chexpert-densenet121")
        _PIPELINE = pipeline("image-classification", model=model)
        return _PIPELINE
    except Exception:
        return None

def analyze_image(image_file: Any):
    image_file.seek(0)
    image = Image.open(image_file).convert("RGB")

    pipe = _load_pipeline()
    if pipe is None:
        # Safe fallback: make the application usable while clearly marking the output.
        return [
            {
                "name": "model_unavailable",
                "probability": 0.0,
                "status": "unavailable",
                "explanation": (
                    "The live Hugging Face vision model could not be loaded. "
                    "Configure HF_TOKEN/model access and restart the app."
                ),
                "source": "vision"
            }
        ]

    outputs = pipe(image)
    findings = []
    for item in outputs:
        label = str(item.get("label", "")).lower().replace(" ", "_")
        score = float(item.get("score", 0.0))
        findings.append({
            "name": label,
            "probability": score,
            "status": "elevated" if score >= 0.5 else "low",
            "explanation": "Pretrained chest-X-ray model output; not a diagnosis.",
            "source": "vision"
        })
    return sorted(findings, key=lambda x: x["probability"], reverse=True)[:8]

import re
import os

COMMON = {
    "opacity": ("present", "An area that looks different from surrounding lung tissue on the X-ray."),
    "airspace opacity": ("present", "An area of the lung appears different from surrounding lung tissue."),
    "pleural effusion": ("present", "Extra fluid around the lung."),
    "cardiomegaly": ("present", "The heart appears larger than usual."),
    "pneumothorax": ("present", "Air is present in the space around the lung."),
    "consolidation": ("present", "An area of lung appears filled or denser than usual."),
    "edema": ("present", "Fluid-related changes in the lungs."),
}

def _status_for(text: str, term: str) -> str:
    lower = text.lower()
    patterns = [
        f"no {term}",
        f"without {term}",
        f"negative for {term}",
        f"absence of {term}",
    ]
    return "absent" if any(p in lower for p in patterns) else "present"

def extract_findings(report_text: str):
    text = " ".join(report_text.split())
    findings = []
    lower = text.lower()

    for term, (default_status, explanation) in COMMON.items():
        if term in lower:
            status = _status_for(lower, term)
            location = None
            if "right lower" in lower and term in lower:
                location = "right lower lung"
            evidence_match = re.search(r"[^.]{0,80}" + re.escape(term) + r"[^.]{0,100}", lower)
            evidence = evidence_match.group(0).strip() if evidence_match else term
            findings.append({
                "name": term.replace(" ", "_"),
                "status": status,
                "location": location,
                "evidence": evidence,
                "plain_language": explanation,
            })

    if not findings:
        try:
            from transformers import pipeline
            model = os.getenv("NLP_MODEL", "d4data/biomedical-ner-all")
            ner = pipeline("token-classification", model=model, aggregation_strategy="simple")
            entities = ner(text[:4000])
            for entity in entities[:20]:
                word = entity.get("word", "").strip()
                if word:
                    findings.append({
                        "name": word.lower().replace(" ", "_"),
                        "status": "uncertain",
                        "location": None,
                        "evidence": word,
                        "plain_language": "A biomedical term identified in the report."
                    })
        except Exception:
            pass

    return findings or [{
        "name": "no_normalized_findings",
        "status": "uncertain",
        "location": None,
        "evidence": "No supported finding pattern was identified.",
        "plain_language": "The prototype could not normalize a specific finding from this report."
    }]

def compare_findings(vision, report):
    report_map = {r["name"]: r for r in report}
    comparisons = []

    for v in vision:
        name = v["name"]
        if name == "model_unavailable":
            continue
        prob = float(v.get("probability", 0.0))
        image_positive = prob >= 0.5

        if name not in report_map:
            comparisons.append({
                "finding": name,
                "image": f"{prob:.0%}",
                "image_probability": prob,
                "report": "not_mentioned",
                "relationship": "not_mentioned",
            })
            continue

        r = report_map[name]
        report_status = r.get("status", "uncertain")

        if report_status == "uncertain":
            relationship = "uncertain"
        elif (image_positive and report_status == "present") or (
            not image_positive and report_status == "absent"
        ):
            relationship = "consistent"
        else:
            relationship = "potential_difference"

        comparisons.append({
            "finding": name,
            "image": f"{prob:.0%}",
            "image_probability": prob,
            "report": report_status,
            "relationship": relationship,
        })

    return comparisons

import os

def generate_explanation(structured):
    api_key = os.getenv("LLM_API_KEY")
    if not api_key:
        return _deterministic_summary(structured)

    # Provider-agnostic placeholder: configure a supported provider here.
    # Keeping this isolated makes the rest of the application provider-independent.
    try:
        provider = os.getenv("LLM_PROVIDER", "").lower()
        if provider == "openai":
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            prompt = f"""
You are explaining a radiology analysis for a general audience.
The clinician report is the primary clinical source. AI vision outputs
are predictions, not diagnoses.

Structured information:
{structured}

Write a concise explanation in plain language. Do not invent findings,
give treatment advice, or override the report. Clearly distinguish
report findings from AI observations.
"""
            response = client.chat.completions.create(
                model=os.getenv("LLM_MODEL", "gpt-5-mini"),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            return response.choices[0].message.content.strip()
    except Exception:
        pass

    return _deterministic_summary(structured)

def _deterministic_summary(structured):
    report = structured.get("report_findings", [])
    comparisons = structured.get("comparisons", [])

    present = [
        r["name"].replace("_", " ")
        for r in report if r.get("status") == "present"
    ]
    absent = [
        r["name"].replace("_", " ")
        for r in report if r.get("status") == "absent"
    ]
    consistent = [
        c["finding"].replace("_", " ")
        for c in comparisons if c["relationship"] == "consistent"
    ]

    parts = []
    if present:
        parts.append("The written report describes: " + ", ".join(present) + ".")
    if absent:
        parts.append("The report specifically describes as absent: " + ", ".join(absent) + ".")
    if consistent:
        parts.append(
            "The prototype's image-model outputs are consistent with the written "
            "report for: " + ", ".join(consistent) + "."
        )
    if not parts:
        parts.append(
            "The prototype could not generate a detailed plain-language summary "
            "from the supplied information."
        )

    parts.append(
        "This is an AI-assisted explanation for educational use. "
        "It is not a medical diagnosis and should not be used for treatment decisions."
    )
    return " ".join(parts)

import uuid

def get_demo_case():
    study_id = "DEMO-" + uuid.uuid4().hex[:6].upper()
    vision = [
        {"name": "opacity", "probability": 0.78, "status": "elevated",
         "explanation": "The pretrained image model assigned a relatively high probability to an opacity-related finding.", "source": "vision"},
        {"name": "pleural_effusion", "probability": 0.10, "status": "low",
         "explanation": "The pretrained image model assigned a low probability to pleural effusion.", "source": "vision"},
        {"name": "cardiomegaly", "probability": 0.21, "status": "low",
         "explanation": "The pretrained image model assigned a low-to-moderate probability to cardiomegaly.", "source": "vision"},
    ]
    report = [
        {"name": "opacity", "status": "present", "location": "right lower lung",
         "evidence": "Patchy right lower lobe airspace opacity is described."},
        {"name": "pleural_effusion", "status": "absent", "location": None,
         "evidence": "No pleural effusion is described."},
    ]
    comparisons = [
        {"finding": "opacity", "image": "78%", "image_probability": 0.78,
         "report": "present", "relationship": "consistent"},
        {"finding": "pleural_effusion", "image": "10%", "image_probability": 0.10,
         "report": "absent", "relationship": "consistent"},
        {"finding": "cardiomegaly", "image": "21%", "image_probability": 0.21,
         "report": "not_mentioned", "relationship": "not_mentioned"},
    ]
    return {
        "result": {
            "study_id": study_id,
            "demo": True,
            "summary": (
                "The example report describes an area of opacity in the lower part "
                "of the right lung and does not describe fluid around the lungs. "
                "The example image-model outputs are broadly consistent with those "
                "documented findings. These values are illustrative and are not a diagnosis."
            ),
            "vision_findings": vision,
            "report_findings": report,
            "comparisons": comparisons,
            "medical_terms": [
                {"term": "Opacity", "plain_language": "An area that looks different from surrounding lung tissue on the X-ray."},
                {"term": "Pleural effusion", "plain_language": "Extra fluid around the lung."},
                {"term": "Cardiomegaly", "plain_language": "The heart appears larger than usual."},
            ],
            "limitations": [
                "Demo output is illustrative.",
                "Model probabilities are not clinical certainty.",
                "This project is not clinically validated."
            ],
            "models": {
                "vision": "itsomk/chexpert-densenet121",
                "nlp": "d4data/biomedical-ner-all",
                "explanation": "Configurable LLM API"
            }
        }
    }

def extract_pdf_text(uploaded_file):
    try:
        import fitz
        data = uploaded_file.read()
        doc = fitz.open(stream=data, filetype="pdf")
        return "\n".join(page.get_text() for page in doc).strip()
    except Exception:
        return ""

import os
import uuid
from typing import Any


def analyze_case(image_file: Any, report_text: str) -> dict:
    study_id = "STUDY-" + uuid.uuid4().hex[:8].upper()
    vision = analyze_image(image_file)
    report = extract_findings(report_text)
    comparisons = compare_findings(vision, report)

    structured = {
        "study_id": study_id,
        "vision_findings": vision,
        "report_findings": report,
        "comparisons": comparisons,
    }

    summary = generate_explanation(structured)

    terms = []
    seen = set()
    for item in report:
        name = item["name"].replace("_", " ").title()
        if name not in seen:
            seen.add(name)
            terms.append({
                "term": name,
                "plain_language": item.get(
                    "plain_language",
                    "A medical finding described in the radiology report."
                )
            })

    return {
        **structured,
        "demo": False,
        "summary": summary,
        "medical_terms": terms,
        "limitations": [
            "AI observations are not diagnoses.",
            "The clinician/radiology report is the primary clinical source.",
            "This prototype has not undergone clinical validation."
        ],
        "models": {
            "vision": os.getenv("VISION_MODEL", "itsomk/chexpert-densenet121"),
            "nlp": os.getenv("NLP_MODEL", "d4data/biomedical-ner-all"),
            "explanation": os.getenv("LLM_MODEL", "configurable")
        }
    }

import json
from pathlib import Path

import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Multimodal Medical Report Visualizer",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Styling ----------
st.markdown("""
<style>
.block-container {max-width: 1400px; padding-top: 2rem; padding-bottom: 3rem;}
.hero {padding: 1.4rem 1.6rem; border: 1px solid rgba(128,128,128,.22);
       border-radius: 18px; background: linear-gradient(135deg, rgba(37,99,235,.10), rgba(14,165,233,.05));}
.small-muted {color: #6b7280; font-size: .9rem;}
.metric-card {padding: 1rem; border: 1px solid rgba(128,128,128,.20);
              border-radius: 14px; background: rgba(128,128,128,.04);}
.status-good {color: #15803d; font-weight: 700;}
.status-warn {color: #b45309; font-weight: 700;}
.status-info {color: #2563eb; font-weight: 700;}
.disclaimer {padding: 1rem; border-radius: 12px; background: rgba(245,158,11,.10);
             border: 1px solid rgba(245,158,11,.25);}
</style>
""", unsafe_allow_html=True)

# ---------- Helpers ----------
def render_finding_card(f):
    name = f.get("name", "Unknown finding").replace("_", " ").title()
    prob = f.get("probability")
    status = f.get("status", "unknown").replace("_", " ").title()
    explanation = f.get("explanation", "")
    location = f.get("location")
    source = f.get("source", "")

    with st.container(border=True):
        c1, c2 = st.columns([3, 1])
        with c1:
            st.markdown(f"### {name}")
            st.caption(f"{source.title()} model" if source else "")
        with c2:
            if prob is not None:
                st.metric("Probability", f"{prob:.0%}")
            else:
                st.metric("Status", status)
        if prob is not None:
            st.progress(max(0.0, min(1.0, float(prob))))
        st.write(explanation)
        if location:
            st.caption(f"Location: {location}")

def make_export(result):
    payload = {
        "study_id": result["study_id"],
        "summary": result["summary"],
        "vision_findings": result["vision_findings"],
        "report_findings": result["report_findings"],
        "comparisons": result["comparisons"],
        "medical_terms": result["medical_terms"],
        "limitations": result["limitations"],
        "models": result["models"],
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)

# ---------- Sidebar ----------
st.sidebar.title("🩻 Medical Visualizer")
page = st.sidebar.radio(
    "Navigate",
    ["Analyze", "History", "Evaluation", "Models & Methodology", "About"],
)

if "history" not in st.session_state:
    st.session_state.history = []

# ---------- Analyze ----------
if page == "Analyze":
    st.markdown("""
    <div class="hero">
      <h1>Multimodal Medical Report Visualizer</h1>
      <p>Explore how pretrained computer-vision and biomedical NLP models
      can connect chest X-ray observations with radiology reports and
      produce patient-friendly explanations.</p>
    </div>
    """, unsafe_allow_html=True)

    st.write("")
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("1. Chest X-ray")
        xray = st.file_uploader(
            "Upload PNG/JPG/JPEG",
            type=["png", "jpg", "jpeg"],
            key="xray",
        )
        if xray:
            st.image(xray, caption="Uploaded X-ray", use_container_width=True)

    with c2:
        st.subheader("2. Radiology report")
        report_file = st.file_uploader(
            "Upload PDF/TXT",
            type=["pdf", "txt"],
            key="report",
        )
        report_text = st.text_area(
            "Or paste/edit the report",
            height=220,
            placeholder="Paste the radiology report here...",
        )

        if report_file and report_file.type == "application/pdf":
            extracted = extract_pdf_text(report_file)
            if extracted and not report_text.strip():
                report_text = extracted
        elif report_file:
            try:
                if not report_text.strip():
                    report_text = report_file.read().decode("utf-8", errors="ignore")
            except Exception:
                st.warning("Could not read the text file.")

        if report_text.strip():
            st.text_area(
                "Extracted / editable report",
                value=report_text,
                height=180,
                key="report_preview",
            )

    st.divider()

    d1, d2, d3 = st.columns([1, 1, 2])
    with d1:
        demo = st.button("🎬 Load Demo Case", use_container_width=True)
    with d2:
        analyze = st.button("🔬 Analyze Study", type="primary", use_container_width=True)
    with d3:
        st.markdown(
            '<div class="small-muted">Educational/research prototype — '
            'not a diagnostic system.</div>',
            unsafe_allow_html=True,
        )

    if demo:
        st.session_state.demo_case = get_demo_case()
        st.session_state.result = st.session_state.demo_case["result"]
        st.rerun()

    if analyze:
        if not xray:
            st.error("Please upload a chest X-ray.")
        elif not report_text.strip():
            st.error("Please upload or paste a radiology report.")
        else:
            with st.status("Running multimodal analysis...", expanded=True) as status:
                st.write("✓ Image loaded")
                st.write("✓ Report extracted")
                result = analyze_case(xray, report_text)
                st.write("✓ Vision inference")
                st.write("✓ Biomedical NLP")
                st.write("✓ Finding normalization")
                st.write("✓ Image/report comparison")
                st.write("✓ Explanation generation")
                status.update(label="Analysis complete", state="complete")
            st.session_state.result = result
            st.session_state.history.insert(0, result)
            st.rerun()

    result = st.session_state.get("result")
    if result:
        st.divider()
        st.header(f"Study {result['study_id']}")
        if result.get("demo"):
            st.info("Demo mode: these results are illustrative and are not from a live clinical case.")

        # Summary
        st.subheader("Plain-language summary")
        st.info(result["summary"])

        # Vision findings
        st.subheader("AI visual findings")
        cols = st.columns(3)
        for i, finding in enumerate(result["vision_findings"][:6]):
            with cols[i % 3]:
                render_finding_card(finding)

        # Report findings
        st.subheader("Radiology report findings")
        for finding in result["report_findings"]:
            with st.container(border=True):
                name = finding["name"].replace("_", " ").title()
                st.markdown(f"### {name}")
                st.write(f"**Status:** {finding['status'].replace('_', ' ').title()}")
                if finding.get("location"):
                    st.write(f"**Location:** {finding['location']}")
                if finding.get("evidence"):
                    st.caption(f"Report evidence: {finding['evidence']}")

        # Comparison
        st.subheader("Image ↔ report comparison")
        comparison_rows = []
        for c in result["comparisons"]:
            comparison_rows.append({
                "Finding": c["finding"].replace("_", " ").title(),
                "Image model": c["image"],
                "Report": c["report"].replace("_", " ").title(),
                "Relationship": c["relationship"].replace("_", " ").title(),
            })
        st.dataframe(pd.DataFrame(comparison_rows), use_container_width=True, hide_index=True)

        # Medical terms
        st.subheader("Medical terminology")
        for term in result["medical_terms"]:
            with st.expander(term["term"]):
                st.write(term["plain_language"])

        # Chart
        st.subheader("Model probability profile")
        chart_df = pd.DataFrame([
            {"Finding": x["name"].replace("_", " ").title(), "Probability": x["probability"]}
            for x in result["vision_findings"] if x.get("probability") is not None
        ])
        if not chart_df.empty:
            st.bar_chart(chart_df.set_index("Finding"))

        # Export
        st.subheader("Export")
        st.download_button(
            "Download JSON analysis",
            data=make_export(result),
            file_name=f"{result['study_id']}.json",
            mime="application/json",
        )

        st.markdown(
            '<div class="disclaimer"><strong>Research prototype:</strong> '
            'AI observations are model outputs, not diagnoses. The clinician/radiology '
            'report is the primary clinical source. Do not use this application for '
            'treatment or medical decision-making.</div>',
            unsafe_allow_html=True,
        )

# ---------- History ----------
elif page == "History":
    st.title("Study History")
    history = st.session_state.history
    if not history:
        st.info("No analyses in this browser session yet. Try the Demo Case.")
    else:
        for r in history:
            with st.expander(f"{r['study_id']} — {r['summary'][:90]}"):
                st.write(r["summary"])
                st.write(f"Comparisons: {len(r['comparisons'])}")
                st.download_button(
                    "Download JSON",
                    make_export(r),
                    file_name=f"{r['study_id']}.json",
                    mime="application/json",
                    key=f"download_{r['study_id']}",
                )

# ---------- Evaluation ----------
elif page == "Evaluation":
    st.title("Evaluation")
    st.write(
        "This page is designed for a small, transparent research evaluation. "
        "Do not report fabricated metrics."
    )
    demo = get_demo_case()
    rows = [
        {
            "Finding": c["finding"].replace("_", " ").title(),
            "Image probability": c["image_probability"],
            "Report status": c["report"],
            "Relationship": c["relationship"].replace("_", " ").title(),
        }
        for c in demo["result"]["comparisons"]
    ]
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)
    if not df.empty:
        consistent = (df["Relationship"] == "Consistent").sum()
        st.metric("Demo-case consistent relationships", f"{consistent}/{len(df)}")
    st.caption(
        "The demo evaluation is illustrative. For a real evaluation, populate "
        "evaluation/ with an appropriately licensed/de-identified dataset and "
        "compute metrics from actual observations."
    )

# ---------- Models ----------
elif page == "Models & Methodology":
    st.title("Models & Methodology")
    st.markdown("""
### Computer vision
A pretrained chest-X-ray classifier is used for image-level finding probabilities.
The model is not trained by this project.

### Biomedical NLP
A pretrained biomedical named-entity model is used to identify medical concepts
from the written report. A lightweight normalization/negation layer maps common
radiology expressions into canonical findings.

### Multimodal fusion
The project compares the two independently produced representations:

`image findings + report findings → relationship`

Possible relationships:

- Consistent
- Potential difference
- Not mentioned
- Uncertain

### Generative explanation
An optional LLM API converts structured findings into plain-language explanations.
The LLM is explicitly instructed not to diagnose, invent findings, or override
the clinician report.

### Important
Model probabilities are not clinical certainty. This is an educational/research
prototype and has not undergone clinical validation.
""")
    st.code("""
X-ray
  ↓
Image preprocessing
  ↓
Pretrained vision model
  ↓
Image findings

Radiology report
  ↓
PDF/text extraction
  ↓
Biomedical NLP + negation
  ↓
Report findings

Image findings + report findings
  ↓
Fusion / comparison
  ↓
LLM explanation
  ↓
Interactive dashboard
""")

# ---------- About ----------
else:
    st.title("About the Project")
    st.markdown("""
## Research question

Can pretrained computer-vision and biomedical NLP models be combined to connect
chest X-ray observations with written radiology findings and produce a clearer,
patient-friendly explanation?

## Why this project exists

Radiology reports contain specialized terminology. At the same time, modern
vision models can extract structured signals from images. This project explores
how the two modalities can be represented separately and then compared.

## Responsible AI

This project does not replace a radiologist or clinician. It is not a medical
device and is not clinically validated.

Do not upload real patient-identifying information.

## Technology

Python · Streamlit · PyTorch · Hugging Face Transformers · Pillow · OpenCV ·
PyMuPDF · Pandas · Plotly · LLM API
""")