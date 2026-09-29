import io
import os
import re
import json
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Multimodal Radiology Analysis",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# APPLICATION CONSTANTS
# ============================================================

VISION_MODEL = os.getenv(
    "VISION_MODEL",
    "itsomk/chexpert-densenet121"
)

NLP_MODEL = os.getenv(
    "NLP_MODEL",
    "d4data/biomedical-ner-all"
)

CHEXPERT_LABELS = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]

# These are display/exploration thresholds.
# They are NOT clinical decision thresholds.
DISPLAY_THRESHOLD = 0.20

# ============================================================
# CLINICAL FINDING KNOWLEDGE
# ============================================================

FINDING_INFO = {
    "No Finding": {
        "plain": "The image model did not identify a strong pattern corresponding to the listed chest abnormalities.",
        "category": "General",
    },
    "Enlarged Cardiomediastinum": {
        "plain": "The model identified a pattern that may correspond to increased width or size of structures in the central chest.",
        "category": "Cardiac / Mediastinal",
    },
    "Cardiomegaly": {
        "plain": "The model identified a pattern that may correspond to an enlarged cardiac silhouette.",
        "category": "Cardiac",
    },
    "Lung Opacity": {
        "plain": "The model identified a region of increased lung density or whiteness.",
        "category": "Lung",
    },
    "Lung Lesion": {
        "plain": "The model identified a pattern that may correspond to a focal abnormal area in the lung.",
        "category": "Lung",
    },
    "Edema": {
        "plain": "The model identified a pattern that can be associated with fluid-related changes in the lungs.",
        "category": "Lung",
    },
    "Consolidation": {
        "plain": "The model identified a pattern of increased lung density that can occur with consolidation.",
        "category": "Lung",
    },
    "Pneumonia": {
        "plain": "The model identified an image pattern that can occur in pneumonia. This model score alone cannot establish that pneumonia is present.",
        "category": "Lung",
    },
    "Atelectasis": {
        "plain": "The model identified a pattern that can be associated with partial collapse or reduced expansion of part of the lung.",
        "category": "Lung",
    },
    "Pneumothorax": {
        "plain": "The model identified a pattern that can be associated with air outside the lung within the chest cavity.",
        "category": "Pleural",
    },
    "Pleural Effusion": {
        "plain": "The model identified a pattern that can be associated with fluid around the lung.",
        "category": "Pleural",
    },
    "Pleural Other": {
        "plain": "The model identified a pattern involving the pleural region that does not specifically correspond to pleural effusion.",
        "category": "Pleural",
    },
    "Fracture": {
        "plain": "The model identified a pattern that may correspond to a bone abnormality or fracture.",
        "category": "Musculoskeletal",
    },
    "Support Devices": {
        "plain": "The model identified a pattern that may correspond to a medical device visible on the radiograph.",
        "category": "Devices",
    },
}

# ============================================================
# REPORT TERMINOLOGY
# ============================================================

REPORT_TERMS = {
    "cardiomegaly": "Cardiomegaly",
    "enlarged heart": "Cardiomegaly",
    "cardiac enlargement": "Cardiomegaly",

    "pleural effusion": "Pleural Effusion",
    "pleural fluid": "Pleural Effusion",
    "effusion": "Pleural Effusion",

    "pneumothorax": "Pneumothorax",
    "collapsed lung": "Pneumothorax",

    "pneumonia": "Pneumonia",

    "lung opacity": "Lung Opacity",
    "pulmonary opacity": "Lung Opacity",
    "airspace opacity": "Lung Opacity",
    "air space opacity": "Lung Opacity",

    "consolidation": "Consolidation",
    "airspace consolidation": "Consolidation",
    "air space consolidation": "Consolidation",

    "atelectasis": "Atelectasis",
    "subsegmental atelectasis": "Atelectasis",
    "basilar atelectasis": "Atelectasis",
    "linear atelectasis": "Atelectasis",

    "pulmonary edema": "Edema",
    "interstitial edema": "Edema",
    "edema": "Edema",

    "lung lesion": "Lung Lesion",
    "pulmonary lesion": "Lung Lesion",
    "lung mass": "Lung Lesion",
    "pulmonary mass": "Lung Lesion",
    "nodule": "Lung Lesion",
    "pulmonary nodule": "Lung Lesion",

    "fracture": "Fracture",
    "rib fracture": "Fracture",

    "support device": "Support Devices",
    "support devices": "Support Devices",
    "central line": "Support Devices",
    "endotracheal tube": "Support Devices",
    "feeding tube": "Support Devices",
}

NEGATION_PATTERNS = [
    r"\bno\b",
    r"\bwithout\b",
    r"\bnegative for\b",
    r"\babsence of\b",
    r"\babsent\b",
    r"\bnot seen\b",
    r"\bnot identified\b",
    r"\bno evidence of\b",
    r"\bwithout evidence of\b",
    r"\bfree of\b",
]

UNCERTAINTY_PATTERNS = [
    r"\bpossible\b",
    r"\bpossibly\b",
    r"\bmay represent\b",
    r"\bmay reflect\b",
    r"\bcannot exclude\b",
    r"\bcannot rule out\b",
    r"\bsuspicious for\b",
    r"\bquestion of\b",
    r"\bquestionable\b",
    r"\bconsider\b",
    r"\bperhaps\b",
]

# ============================================================
# CSS
# ============================================================

st.markdown(
    """
<style>

:root {
    --primary: #1f5f9c;
    --primary-dark: #164873;
    --text: #172033;
    --muted: #657084;
    --border: #dce2ea;
    --surface: #ffffff;
    --surface-soft: #f6f8fb;
    --success: #2d6a4f;
    --warning: #946200;
    --danger: #9b3434;
}

html, body, [class*="css"] {
    font-family:
        Inter,
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        sans-serif;
}

.stApp {
    background: #ffffff;
    color: var(--text);
}

.block-container {
    max-width: 1380px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}

h1, h2, h3, h4 {
    color: var(--text);
    letter-spacing: -0.02em;
}

h1 {
    font-size: 2.4rem !important;
    font-weight: 700 !important;
}

h2 {
    font-size: 1.55rem !important;
    font-weight: 650 !important;
}

h3 {
    font-size: 1.15rem !important;
    font-weight: 650 !important;
}

p {
    color: var(--text);
    line-height: 1.65;
}

[data-testid="stSidebar"] {
    background: #f7f9fc;
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] * {
    color: var(--text);
}

.app-kicker {
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--primary);
    margin-bottom: 0.45rem;
}

.app-title {
    font-size: 2.45rem;
    line-height: 1.12;
    font-weight: 750;
    letter-spacing: -0.04em;
    margin-bottom: 0.6rem;
}

.app-subtitle {
    color: var(--muted);
    font-size: 1.02rem;
    max-width: 900px;
    line-height: 1.65;
}

.section-label {
    color: var(--primary);
    font-size: 0.75rem;
    font-weight: 750;
    letter-spacing: 0.10em;
    text-transform: uppercase;
    margin-top: 1.5rem;
    margin-bottom: 0.35rem;
}

.section-description {
    color: var(--muted);
    font-size: 0.92rem;
    margin-bottom: 1rem;
}

.surface {
    border: 1px solid var(--border);
    background: var(--surface);
    border-radius: 12px;
    padding: 1.25rem;
    margin-bottom: 1rem;
}

.soft-surface {
    border: 1px solid var(--border);
    background: var(--surface-soft);
    border-radius: 12px;
    padding: 1.15rem;
    margin-bottom: 1rem;
}

.finding-card {
    border: 1px solid var(--border);
    border-radius: 12px;
    background: white;
    padding: 1.15rem;
    min-height: 190px;
    margin-bottom: 1rem;
}

.finding-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--text);
    margin-bottom: 0.4rem;
}

.finding-score {
    font-size: 1.8rem;
    font-weight: 650;
    color: var(--text);
}

.finding-caption {
    color: var(--muted);
    font-size: 0.83rem;
    line-height: 1.5;
    margin-top: 0.6rem;
}

.evidence-box {
    background: #f8fafc;
    border-left: 3px solid var(--primary);
    padding: 0.9rem 1rem;
    border-radius: 0 8px 8px 0;
    margin: 0.6rem 0;
}

.evidence-label {
    color: var(--muted);
    font-size: 0.74rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

.evidence-text {
    color: var(--text);
    font-size: 0.9rem;
    margin-top: 0.25rem;
}

.status-consistent {
    color: var(--success);
    font-weight: 700;
}

.status-difference {
    color: var(--warning);
    font-weight: 700;
}

.status-report {
    color: #315f91;
    font-weight: 700;
}

.status-uncertain {
    color: #805d19;
    font-weight: 700;
}

.status-neutral {
    color: var(--muted);
    font-weight: 650;
}

.metric-label {
    color: var(--muted);
    font-size: 0.77rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 700;
}

.metric-value {
    color: var(--text);
    font-size: 1.6rem;
    font-weight: 700;
}

.disclaimer {
    border: 1px solid #e2e5e9;
    background: #fafafa;
    border-radius: 10px;
    padding: 1rem 1.1rem;
    color: #555f6d;
    font-size: 0.83rem;
    line-height: 1.6;
    margin-top: 1rem;
}

.small-note {
    color: var(--muted);
    font-size: 0.82rem;
    line-height: 1.55;
}

.method-box {
    border-top: 1px solid var(--border);
    padding-top: 1rem;
    margin-top: 1rem;
}

[data-testid="stFileUploader"] {
    border: 1px solid var(--border);
    border-radius: 10px;
}

.stButton > button {
    border-radius: 8px;
    border: 1px solid #cbd4df;
    font-weight: 600;
}

.stButton > button[kind="primary"] {
    background: var(--primary);
    border-color: var(--primary);
}

.stButton > button[kind="primary"]:hover {
    background: var(--primary-dark);
    border-color: var(--primary-dark);
}

[data-testid="stMetric"] {
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.8rem;
    background: white;
}

hr {
    border: none;
    border-top: 1px solid var(--border);
    margin: 2rem 0;
}

</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# SESSION STATE
# ============================================================

if "history" not in st.session_state:
    st.session_state.history = []

if "last_analysis" not in st.session_state:
    st.session_state.last_analysis = None

# ============================================================
# UTILITY FUNCTIONS
# ============================================================


def safe_float(value):
    try:
        return float(value)
    except Exception:
        return 0.0


def format_percentage(value):
    return f"{safe_float(value) * 100:.1f}%"


def clean_text(text):
    if not text:
        return ""

    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_term(term):
    if not term:
        return None

    term = term.lower().strip()

    if term in REPORT_TERMS:
        return REPORT_TERMS[term]

    return None


# ============================================================
# MODEL LOADING
# ============================================================


@st.cache_resource(show_spinner=False)
def load_vision_model():

    import torch
    import torchvision.models as models
    import torchvision.transforms as transforms
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    class DenseNet121CheXpert(torch.nn.Module):

        def __init__(self, num_labels=14):
            super().__init__()

            self.densenet = models.densenet121(
                weights=None
            )

            num_features = (
                self.densenet.classifier.in_features
            )

            self.densenet.classifier = torch.nn.Linear(
                num_features,
                num_labels
            )

        def forward(self, x):
            return self.densenet(x)

    model = DenseNet121CheXpert(
        num_labels=len(CHEXPERT_LABELS)
    )

    checkpoint_path = hf_hub_download(
        repo_id=VISION_MODEL,
        filename="pytorch_model.safetensors",
    )

    state_dict = load_file(checkpoint_path)

    model.load_state_dict(
        state_dict,
        strict=False
    )

    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[
                    0.485,
                    0.456,
                    0.406,
                ],
                std=[
                    0.229,
                    0.224,
                    0.225,
                ],
            ),
        ]
    )

    return model, transform


# ============================================================
# X-RAY ANALYSIS
# ============================================================


def analyze_xray(image_bytes):

    import torch

    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    model, transform = load_vision_model()

    tensor = transform(image).unsqueeze(0)

    with torch.inference_mode():

        logits = model(tensor)

        probabilities = torch.sigmoid(
            logits
        )[0].cpu().numpy()

    results = []

    for label, probability in zip(
        CHEXPERT_LABELS,
        probabilities
    ):

        results.append(
            {
                "finding": label,
                "probability": float(probability),
            }
        )

    results.sort(
        key=lambda x: x["probability"],
        reverse=True
    )

    return results


# ============================================================
# PDF EXTRACTION
# ============================================================


def extract_pdf_text(file_bytes):

    import fitz

    text_parts = []

    document = fitz.open(
        stream=file_bytes,
        filetype="pdf"
    )

    for page in document:
        text_parts.append(
            page.get_text()
        )

    document.close()

    return "\n".join(text_parts)


# ============================================================
# REPORT PARSING
# ============================================================


def get_local_context(text, start, end, window=90):

    left = max(0, start - window)
    right = min(
        len(text),
        end + window
    )

    return text[left:right]


def context_has_negation(context):

    context_lower = context.lower()

    for pattern in NEGATION_PATTERNS:

        if re.search(
            pattern,
            context_lower
        ):
            return True

    return False


def context_has_uncertainty(context):

    context_lower = context.lower()

    for pattern in UNCERTAINTY_PATTERNS:

        if re.search(
            pattern,
            context_lower
        ):
            return True

    return False


def classify_report_mention(
    context
):

    if context_has_negation(
        context
    ):
        return "NEGATED"

    if context_has_uncertainty(
        context
    ):
        return "UNCERTAIN"

    return "MENTIONED"


def parse_report(text):

    text = clean_text(text)

    findings = {}

    for term, canonical in REPORT_TERMS.items():

        pattern = re.escape(term)

        for match in re.finditer(
            pattern,
            text,
            flags=re.IGNORECASE
        ):

            context = get_local_context(
                text,
                match.start(),
                match.end()
            )

            status = classify_report_mention(
                context
            )

            if canonical not in findings:

                findings[canonical] = {
                    "finding": canonical,
                    "status": status,
                    "evidence": context.strip(),
                }

            else:

                current = findings[canonical]

                # Prefer explicit negation/uncertainty
                # over generic mention if encountered.
                priority = {
                    "NEGATED": 3,
                    "UNCERTAIN": 2,
                    "MENTIONED": 1,
                }

                if (
                    priority.get(status, 0)
                    >
                    priority.get(
                        current["status"],
                        0
                    )
                ):

                    findings[canonical] = {
                        "finding": canonical,
                        "status": status,
                        "evidence": context.strip(),
                    }

    return list(findings.values())


# ============================================================
# OPTIONAL BIOMEDICAL NER
# ============================================================


@st.cache_resource(show_spinner=False)
def load_ner_pipeline():

    from transformers import pipeline

    return pipeline(
        "token-classification",
        model=NLP_MODEL,
        aggregation_strategy="simple"
    )


def run_optional_ner(text):

    """
    NER is used only as a supplementary extraction layer.
    The application's normalized finding dictionary remains
    the primary deterministic parser.
    """

    if not text.strip():
        return []

    try:

        ner = load_ner_pipeline()

        entities = ner(
            text[:12000]
        )

        return entities

    except Exception:

        return []


# ============================================================
# CROSS-MODAL COMPARISON
# ============================================================


def build_comparison(
    vision_results,
    report_findings
):

    report_map = {
        item["finding"]: item
        for item in report_findings
    }

    comparison = []

    for item in vision_results:

        finding = item["finding"]
        score = safe_float(
            item["probability"]
        )

        report = report_map.get(
            finding
        )

        if report is None:

            comparison.append(
                {
                    "finding": finding,
                    "image_score": score,
                    "report_status": "NOT_MENTIONED",
                    "relationship": "NOT_MENTIONED",
                    "evidence": "",
                }
            )

            continue

        report_status = report[
            "status"
        ]

        if report_status == "MENTIONED":

            relationship = "CONSISTENT"

        elif report_status == "NEGATED":

            if score >= DISPLAY_THRESHOLD:
                relationship = (
                    "POTENTIAL_DIFFERENCE"
                )
            else:
                relationship = (
                    "CONSISTENT"
                )

        elif report_status == "UNCERTAIN":

            relationship = "UNCERTAIN"

        else:

            relationship = "REVIEW"

        comparison.append(
            {
                "finding": finding,
                "image_score": score,
                "report_status": report_status,
                "relationship": relationship,
                "evidence": report.get(
                    "evidence",
                    ""
                ),
            }
        )

    # Add report findings not represented
    # in the top vision results.
    vision_names = {
        item["finding"]
        for item in vision_results
    }

    for report in report_findings:

        if report["finding"] not in vision_names:

            comparison.append(
                {
                    "finding": report["finding"],
                    "image_score": None,
                    "report_status": report[
                        "status"
                    ],
                    "relationship": (
                        "REPORT_ONLY"
                    ),
                    "evidence": report.get(
                        "evidence",
                        ""
                    ),
                }
            )

    return comparison


# ============================================================
# PLAIN LANGUAGE EXPLANATION
# ============================================================


def generate_plain_language_summary(
    vision_findings,
    report_findings,
    comparison
):

    strong_vision = []

    for item in vision_findings:

        score = safe_float(
            item.get(
                "probability",
                item.get(
                    "score",
                    0
                )
            )
        )

        name = (
            item.get("finding")
            or item.get("label")
            or item.get("name")
        )

        if (
            name
            and score >= DISPLAY_THRESHOLD
            and name != "No Finding"
        ):

            strong_vision.append(
                (name, score)
            )

    strong_vision.sort(
        key=lambda x: x[1],
        reverse=True
    )

    paragraphs = []

    if strong_vision:

        paragraphs.append(
            "The image model identified several "
            "radiographic patterns that it considered "
            "worth highlighting."
        )

        for name, score in strong_vision[:5]:

            description = FINDING_INFO.get(
                name,
                {}
            ).get(
                "plain",
                f"The model identified a pattern associated with {name.lower()}."
            )

            paragraphs.append(
                f"**{name} ({format_percentage(score)} model score):** "
                f"{description}"
            )

    else:

        paragraphs.append(
            "The image model did not produce any "
            "higher-scoring finding above the application's "
            "display threshold."
        )

    if report_findings:

        report_items = []

        for item in report_findings[:10]:

            name = item["finding"]
            status = item["status"]

            if status == "MENTIONED":
                report_items.append(
                    f"{name} was mentioned in the report."
                )

            elif status == "NEGATED":
                report_items.append(
                    f"{name} was explicitly described as absent or negative."
                )

            elif status == "UNCERTAIN":
                report_items.append(
                    f"{name} was described with uncertainty."
                )

        if report_items:

            paragraphs.append(
                "The report-processing layer identified "
                "the following clinically relevant statements:"
            )

            paragraphs.extend(
                [
                    f"- {item}"
                    for item in report_items
                ]
            )

    else:

        paragraphs.append(
            "The report-processing layer did not identify "
            "a supported finding in the supplied text."
        )

    meaningful_comparisons = [
        item
        for item in comparison
        if item["relationship"]
        != "NOT_MENTIONED"
    ]

    if meaningful_comparisons:

        paragraphs.append(
            "The multimodal comparison then examined "
            "whether findings extracted from the report "
            "were compatible with the image-model outputs."
        )

        consistent = sum(
            1
            for item in meaningful_comparisons
            if item["relationship"]
            == "CONSISTENT"
        )

        differences = sum(
            1
            for item in meaningful_comparisons
            if item["relationship"]
            == "POTENTIAL_DIFFERENCE"
        )

        uncertain = sum(
            1
            for item in meaningful_comparisons
            if item["relationship"]
            == "UNCERTAIN"
        )

        paragraphs.append(
            f"The current case contains {consistent} "
            f"relationship(s) classified as consistent, "
            f"{differences} potential cross-modal difference(s), "
            f"and {uncertain} uncertain relationship(s)."
        )

    paragraphs.append(
        "These results are outputs of a pretrained research "
        "model and automated text processing. Model scores "
        "are not calibrated clinical probabilities and do "
        "not constitute a medical diagnosis."
    )

    paragraphs.append(
        "The supplied radiology report should remain the "
        "primary clinical source for interpreting the examination."
    )

    return "\n\n".join(
        paragraphs
    )


# ============================================================
# DEMO DATA
# ============================================================


def get_demo_analysis():

    vision = [
        {
            "finding": "Lung Opacity",
            "probability": 0.78,
        },
        {
            "finding": "Pneumonia",
            "probability": 0.67,
        },
        {
            "finding": "Lung Lesion",
            "probability": 0.41,
        },
        {
            "finding": "Pleural Effusion",
            "probability": 0.18,
        },
        {
            "finding": "Cardiomegaly",
            "probability": 0.12,
        },
    ]

    report_text = """
    Frontal chest radiograph.

    There is patchy airspace opacity in the right lower lung.
    The cardiac silhouette is mildly enlarged.
    No pleural effusion or pneumothorax is identified.

    IMPRESSION:
    Right lower lung airspace opacity.
    Mild cardiomegaly.
    No pleural effusion.
    """

    report_findings = parse_report(
        report_text
    )

    comparison = build_comparison(
        vision,
        report_findings
    )

    summary = generate_plain_language_summary(
        vision,
        report_findings,
        comparison
    )

    return {
        "vision": vision,
        "report_findings": report_findings,
        "comparison": comparison,
        "report_text": report_text,
        "summary": summary,
    }


# ============================================================
# DISPLAY HELPERS
# ============================================================
def render_finding_card(finding, score):
    description = FINDING_INFO.get(
        finding,
        {}
    ).get(
        "plain",
        "The model produced a score for this finding."
    )

    with st.container(border=True):
        st.markdown(f"### {finding}")

        st.caption("Research model score")

        st.markdown(
            f"## {format_percentage(score)}"
        )

        st.progress(
            min(max(float(score), 0.0), 1.0)
        )

        st.caption(description)

def render_report_status(status):

    mapping = {
        "MENTIONED": (
            "Mentioned",
            "status-report"
        ),
        "NEGATED": (
            "Negated",
            "status-neutral"
        ),
        "UNCERTAIN": (
            "Uncertain",
            "status-uncertain"
        ),
        "NOT_MENTIONED": (
            "Not mentioned",
            "status-neutral"
        ),
    }

    return mapping.get(
        status,
        (
            status,
            "status-neutral"
        )
    )


def render_relationship(
    relationship
):

    mapping = {
        "CONSISTENT": (
            "Consistent",
            "status-consistent"
        ),
        "POTENTIAL_DIFFERENCE": (
            "Potential difference",
            "status-difference"
        ),
        "UNCERTAIN": (
            "Uncertain",
            "status-uncertain"
        ),
        "REPORT_ONLY": (
            "Report finding",
            "status-report"
        ),
        "NOT_MENTIONED": (
            "Not mentioned",
            "status-neutral"
        ),
    }

    return mapping.get(
        relationship,
        (
            relationship,
            "status-neutral"
        )
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class="app-kicker">
            Independent Research Prototype
        </div>

        <div style="
            font-size:1.25rem;
            font-weight:700;
            margin-bottom:0.4rem;
        ">
            Multimodal Radiology Analysis
        </div>

        <div class="small-note">
            Structured analysis of chest radiographs
            and radiology reports.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    page = st.radio(
        "Application",
        [
            "Analysis",
            "Case History",
            "Evaluation",
            "Methodology",
            "Models",
            "Limitations",
            "About",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    st.markdown(
        """
        <div class="small-note">
        <strong>Research status</strong><br>
        Educational prototype. Not clinically validated.
        </div>
        """,
        unsafe_allow_html=True
    )

# ============================================================
# PAGE: ANALYSIS
# ============================================================

if page == "Analysis":

    st.markdown(
        """
        <div class="app-kicker">
            Multimodal Medical AI
        </div>

        <div class="app-title">
            Multimodal Radiology Analysis
        </div>

        <div class="app-subtitle">
            An independent research prototype that processes
            chest radiographs and radiology reports as separate
            information sources, then examines their relationship
            through structured multimodal analysis.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-label">Case input</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="section-description">
            Provide both components of a case for the complete
            multimodal workflow. The radiograph is processed by
            a pretrained chest X-ray classifier, while the report
            is analyzed for findings, negation and uncertainty.
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(
        2,
        gap="large"
    )

    with col1:

        st.markdown(
            "### Chest radiograph"
        )

        xray_file = st.file_uploader(
            "Upload PNG, JPG or JPEG",
            type=[
                "png",
                "jpg",
                "jpeg",
            ],
            key="xray_upload",
        )

        if xray_file:

            image = Image.open(
                xray_file
            ).convert("RGB")

            st.image(
                image,
                caption="Uploaded radiograph",
                use_container_width=True,
            )

            st.markdown(
                f"""
                <div class="small-note">
                    File: {xray_file.name}
                    <br>
                    Dimensions: {image.width} × {image.height} pixels
                </div>
                """,
                unsafe_allow_html=True
            )

    with col2:

        st.markdown(
            "### Radiology report"
        )

        report_file = st.file_uploader(
            "Upload PDF or TXT",
            type=[
                "pdf",
                "txt",
            ],
            key="report_upload",
        )

        pasted_report = st.text_area(
            "Paste or edit the report",
            height=230,
            placeholder=(
                "Example:\n"
                "No pleural effusion. "
                "Patchy right lower lung opacity..."
            ),
            label_visibility="visible",
        )

        extracted_report = ""

        if report_file:

            if report_file.name.lower().endswith(
                ".pdf"
            ):

                try:

                    extracted_report = extract_pdf_text(
                        report_file.getvalue()
                    )

                except Exception as exc:

                    st.error(
                        "The PDF could not be read."
                    )

                    st.exception(exc)

            else:

                try:

                    extracted_report = (
                        report_file
                        .getvalue()
                        .decode(
                            "utf-8",
                            errors="ignore"
                        )
                    )

                except Exception:

                    extracted_report = ""

        report_text = (
            pasted_report.strip()
            if pasted_report.strip()
            else extracted_report.strip()
        )

        if report_text:

            with st.expander(
                "Review extracted report",
                expanded=False
            ):

                st.text_area(
                    "Report text",
                    value=report_text,
                    height=220,
                    key="report_preview",
                )

    st.divider()

    analysis_col1, analysis_col2 = st.columns(
        [1, 4]
    )

    with analysis_col1:

        run_analysis = st.button(
            "Run analysis",
            type="primary",
            use_container_width=True,
        )

    with analysis_col2:

        demo_mode = st.checkbox(
            "Use demonstration case",
            value=False,
        )

    if demo_mode:

        demo = get_demo_analysis()

        st.session_state.last_analysis = {
            "timestamp": datetime.now().isoformat(),
            "source": "Demonstration case",
            **demo,
        }

        st.info(
            "Demonstration mode uses illustrative structured data. "
            "It is not an analysis of a real patient examination."
        )

    elif run_analysis:

        if not xray_file:

            st.error(
                "Please provide a chest radiograph."
            )

        elif not report_text:

            st.error(
                "Please provide a radiology report or report text."
            )

        else:

            progress = st.progress(
                0,
                text="Preparing analysis..."
            )

            try:

                progress.progress(
                    20,
                    text="Processing chest radiograph..."
                )

                image_bytes = (
                    xray_file.getvalue()
                )

                vision_results = analyze_xray(
                    image_bytes
                )

                progress.progress(
                    50,
                    text="Extracting report findings..."
                )

                report_findings = parse_report(
                    report_text
                )

                progress.progress(
                    70,
                    text="Building cross-modal comparison..."
                )

                comparison = build_comparison(
                    vision_results,
                    report_findings
                )

                progress.progress(
                    90,
                    text="Generating structured explanation..."
                )

                summary = generate_plain_language_summary(
                    vision_results,
                    report_findings,
                    comparison
                )

                progress.progress(
                    100,
                    text="Analysis complete."
                )

                analysis = {
                    "timestamp": datetime.now().isoformat(),
                    "source": "User-provided case",
                    "xray_filename": xray_file.name,
                    "vision": vision_results,
                    "report_findings": report_findings,
                    "comparison": comparison,
                    "report_text": report_text,
                    "summary": summary,
                }

                st.session_state.last_analysis = analysis

                st.session_state.history.append(
                    {
                        "timestamp": analysis[
                            "timestamp"
                        ],
                        "xray": xray_file.name,
                        "report": (
                            report_file.name
                            if report_file
                            else "Pasted report"
                        ),
                        "findings": len(
                            vision_results
                        ),
                    }
                )

            except Exception as exc:

                st.error(
                    "The analysis pipeline encountered an error."
                )

                st.exception(exc)

    # ========================================================
    # DISPLAY ANALYSIS
    # ========================================================

    analysis = st.session_state.last_analysis

    if analysis:

        st.divider()

        st.markdown(
            '<div class="section-label">Image analysis</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="section-description">
                The following scores are produced independently
                by the pretrained chest X-ray model. Multiple
                findings can receive high scores because this is
                a multi-label classification problem.
            </div>
            """,
            unsafe_allow_html=True
        )

        vision = analysis["vision"]

        top_results = [
            item
            for item in vision
            if item["probability"]
            >= DISPLAY_THRESHOLD
        ][:9]

        if not top_results:

            st.info(
                "No finding exceeded the application's "
                "display threshold."
            )

        else:

            for start in range(
                0,
                len(top_results),
                3
            ):

                row = top_results[
                    start:start + 3
                ]

                cols = st.columns(
                    len(row),
                    gap="medium"
                )

                for col, item in zip(
                    cols,
                    row
                ):

                    with col:

                        render_finding_card(
                            item["finding"],
                            item["probability"]
                        )

        st.markdown(
            """
            <div class="disclaimer">
            <strong>Interpretation note.</strong>
            These values are research-model outputs rather than
            calibrated clinical probabilities. They should not be
            interpreted as diagnoses or treatment recommendations.
            </div>
            """,
            unsafe_allow_html=True
        )

        # ====================================================
        # REPORT ANALYSIS
        # ====================================================

        st.markdown(
            '<div class="section-label">Report analysis</div>',
            unsafe_allow_html=True
        )

        report_findings = analysis[
            "report_findings"
        ]

        if not report_findings:

            st.info(
                "No supported findings were extracted from the supplied report."
            )

        else:

            report_df = pd.DataFrame(
                report_findings
            )

            report_df = report_df[
                [
                    "finding",
                    "status",
                    "evidence",
                ]
            ]

            report_df.columns = [
                "Finding",
                "Report status",
                "Evidence",
            ]

            st.dataframe(
                report_df,
                use_container_width=True,
                hide_index=True,
            )

        # ====================================================
        # CROSS MODAL
        # ====================================================

        st.markdown(
            '<div class="section-label">Cross-modal analysis</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="section-description">
                The comparison layer does not determine whether a
                clinician or model is correct. It identifies the
                relationship between the image-model output and
                information explicitly extracted from the report.
            </div>
            """,
            unsafe_allow_html=True
        )

        comparison = analysis[
            "comparison"
        ]

        comparison_rows = []

        for item in comparison:

            score = item[
                "image_score"
            ]

            report_status = item[
                "report_status"
            ]

            relationship = item[
                "relationship"
            ]

            comparison_rows.append(
                {
                    "Finding": item[
                        "finding"
                    ],
                    "Image model score": (
                        format_percentage(score)
                        if score is not None
                        else "—"
                    ),
                    "Report": (
                        render_report_status(
                            report_status
                        )[0]
                    ),
                    "Relationship": (
                        render_relationship(
                            relationship
                        )[0]
                    ),
                }
            )

        if comparison_rows:

            comparison_df = pd.DataFrame(
                comparison_rows
            )

            st.dataframe(
                comparison_df,
                use_container_width=True,
                hide_index=True,
            )

            st.markdown(
                """
                <div class="small-note">
                <strong>Relationship definitions:</strong>
                Consistent means the two information sources do not
                show an obvious conflict under the application's
                simple comparison rules. Potential difference means
                the image score and report statement differ enough
                to warrant inspection. It does not mean that either
                source is incorrect.
                </div>
                """,
                unsafe_allow_html=True
            )

        # ====================================================
        # EVIDENCE
        # ====================================================

        st.markdown(
            '<div class="section-label">Report evidence</div>',
            unsafe_allow_html=True
        )

        if report_findings:

            for item in report_findings:

                status_label, status_class = (
                    render_report_status(
                        item["status"]
                    )
                )

                st.markdown(
                    f"""
                    <div class="evidence-box">

                        <div class="evidence-label">
                            {item["finding"]} ·
                            <span class="{status_class}">
                                {status_label}
                            </span>
                        </div>

                        <div class="evidence-text">
                            {item["evidence"]}
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

        # ====================================================
        # PLAIN LANGUAGE
        # ====================================================

        st.markdown(
            '<div class="section-label">Plain-language explanation</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            analysis["summary"]
        )

        # ====================================================
        # VISUAL ANALYTICS
        # ====================================================

        st.markdown(
            '<div class="section-label">Model output profile</div>',
            unsafe_allow_html=True
        )

        chart_df = pd.DataFrame(
            vision
        )

        chart_df = chart_df.sort_values(
            "probability",
            ascending=True
        )

        st.bar_chart(
            chart_df.set_index(
                "finding"
            )[
                "probability"
            ]
        )

        # ====================================================
        # EXPORT
        # ====================================================

        st.markdown(
            '<div class="section-label">Export</div>',
            unsafe_allow_html=True
        )

        export_data = {
            "project": (
                "Multimodal Radiology Analysis"
            ),
            "timestamp": analysis[
                "timestamp"
            ],
            "source": analysis[
                "source"
            ],
            "vision_model": VISION_MODEL,
            "nlp_model": NLP_MODEL,
            "vision_findings": analysis[
                "vision"
            ],
            "report_findings": analysis[
                "report_findings"
            ],
            "cross_modal_comparison": analysis[
                "comparison"
            ],
            "plain_language_summary": analysis[
                "summary"
            ],
            "limitations": [
                "Research prototype.",
                "Not clinically validated.",
                "Model scores are not calibrated clinical probabilities.",
                "Radiology report remains the primary clinical source.",
                "Agreement with a report is not equivalent to ground-truth accuracy.",
            ],
        }

        st.download_button(
            "Download structured analysis",
            data=json.dumps(
                export_data,
                indent=2
            ),
            file_name="radiology_multimodal_analysis.json",
            mime="application/json",
            use_container_width=False,
        )


# ============================================================
# PAGE: CASE HISTORY
# ============================================================

elif page == "Case History":

    st.markdown(
        '<div class="app-kicker">Case management</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="app-title">Case History</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="app-subtitle">
            A session-level record of cases analyzed during the
            current browser session. No persistent patient database
            is implemented in this prototype.
        </div>
        """,
        unsafe_allow_html=True
    )

    if not st.session_state.history:

        st.info(
            "No cases have been analyzed in this session."
        )

    else:

        history_df = pd.DataFrame(
            st.session_state.history
        )

        history_df.columns = [
            "Timestamp",
            "X-ray",
            "Report",
            "Findings processed",
        ]

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True,
        )

    st.markdown(
        """
        <div class="disclaimer">
        <strong>Privacy note.</strong>
        The prototype does not implement persistent patient storage.
        Do not upload identifiable clinical information to the public
        deployment.
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# PAGE: EVALUATION
# ============================================================

elif page == "Evaluation":

    st.markdown(
        '<div class="app-kicker">Research evaluation</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="app-title">Evaluation Framework</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="app-subtitle">
            The evaluation layer separates model behavior,
            report extraction and multimodal comparison instead
            of treating report agreement as clinical accuracy.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        "### 1. Image-model evaluation"
    )

    st.markdown(
        """
        The vision component produces independent scores for
        multiple radiographic findings. Because this is a
        multi-label classifier, the outputs do not form a
        probability distribution that sums to 100 percent.

        Appropriate future evaluation would use an independently
        annotated test set and report class-specific metrics such
        as AUROC, sensitivity, specificity and calibration.
        """
    )

    st.markdown(
        "### 2. Report extraction evaluation"
    )

    st.markdown(
        """
        The text-processing layer is evaluated conceptually along
        three dimensions:

        - finding normalization
        - negation detection
        - uncertainty detection

        For example, "no pleural effusion" should not be converted
        into a positive pleural-effusion finding.
        """
    )

    st.markdown(
        "### 3. Multimodal evaluation"
    )

    st.markdown(
        """
        The comparison layer evaluates whether information from
        the two modalities is compatible under explicit rules.

        A report-model match should not automatically be described
        as a correct prediction. A radiology report is a clinical
        information source, not necessarily an independent
        ground-truth annotation.
        """
    )

    st.markdown(
        "### 4. Reproducible testing protocol"
    )

    st.markdown(
        """
        A stronger future experiment would use paired,
        de-identified radiographs and reports from a public
        research dataset.

        Each case can then be evaluated for:

        1. image-model output
        2. report finding extraction
        3. negation handling
        4. uncertainty handling
        5. cross-modal relationship
        6. human review of disagreements
        """
    )

    st.markdown(
        """
        <div class="disclaimer">
        This prototype does not claim clinical accuracy. No
        performance number is presented here unless it has been
        calculated on an explicitly defined, independently
        evaluated dataset.
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# PAGE: METHODOLOGY
# ============================================================

elif page == "Methodology":

    st.markdown(
        '<div class="app-kicker">System architecture</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="app-title">Methodology</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="app-subtitle">
            The system is designed as a multimodal pipeline in
            which image and language are processed independently
            before their structured outputs are compared.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        "### System pipeline"
    )

    st.code(
        """
Chest X-ray
    |
    v
Image preprocessing
    |
    v
DenseNet-121 chest X-ray classifier
    |
    v
14 multi-label radiographic scores
    |
    +-----------------------------+
                                  |
                                  v
                        Finding normalization
                                  ^
                                  |
Radiology report                 |
    |                             |
    v                             |
PDF/TXT extraction                |
    |                             |
    v                             |
Clinical terminology detection ---+
    |
    v
Negation and uncertainty analysis
    |
    v
Structured report findings
    |
    +-------------+---------------+
                  |
                  v
        Cross-modal comparison
                  |
                  v
        Plain-language explanation
                  |
                  v
        Structured JSON export
        """,
        language="text",
    )

    st.markdown(
        "### Image processing"
    )

    st.markdown(
        """
        The radiograph is converted to RGB and resized to the
        model input resolution. ImageNet normalization is applied
        before inference. The pretrained DenseNet-121 model produces
        independent sigmoid outputs for the supported chest
        radiographic categories.
        """
    )

    st.markdown(
        "### Clinical text processing"
    )

    st.markdown(
        """
        The report is first converted into plain text. A normalized
        terminology layer maps common expressions to canonical
        findings. Local context is then examined for negation and
        uncertainty.

        For example:

        "No pleural effusion"

        becomes:

        Finding: Pleural Effusion  
        Status: NEGATED

        rather than a positive finding.
        """
    )

    st.markdown(
        "### Multimodal fusion"
    )

    st.markdown(
        """
        The system does not concatenate image pixels and text
        embeddings into a single black-box model. Instead, it uses
        structured intermediate representations.

        This design makes the reasoning process inspectable:

        image → finding scores

        report → normalized findings + evidence

        findings → cross-modal relationship

        This is particularly useful for a research prototype
        because each stage can be independently inspected and
        evaluated.
        """
    )

    st.markdown(
        "### Technology stack"
    )

    stack_df = pd.DataFrame(
        [
            ["Python", "Application and analysis pipeline"],
            ["Streamlit", "Interactive research interface"],
            ["PyTorch", "Neural-network inference"],
            ["Torchvision", "DenseNet-121 architecture"],
            ["Hugging Face Hub", "Pretrained model distribution"],
            ["Transformers", "Biomedical NLP"],
            ["PyMuPDF", "PDF text extraction"],
            ["Pillow", "Image processing"],
            ["Pandas", "Structured analysis"],
            ["Plotly / Streamlit charts", "Visualization"],
        ],
        columns=[
            "Technology",
            "Role",
        ],
    )

    st.dataframe(
        stack_df,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# PAGE: MODELS
# ============================================================

elif page == "Models":

    st.markdown(
        '<div class="app-kicker">Machine learning components</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="app-title">Models</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="app-subtitle">
            The prototype combines a pretrained radiographic
            classifier with biomedical language processing.
            No model is trained inside the application.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        "### Vision model"
    )

    st.markdown(
        f"""
        **Model:** `{VISION_MODEL}`

        **Architecture:** DenseNet-121

        **Task:** Multi-label chest X-ray classification

        **Outputs:** 14 radiographic categories

        The model produces independent sigmoid scores for each
        category. These values are presented as model scores rather
        than clinical probabilities.
        """
    )

    st.markdown(
        "### NLP model"
    )

    st.markdown(
        f"""
        **Model:** `{NLP_MODEL}`

        The NLP component is used as a supplementary biomedical
        entity-recognition layer. The primary report interpretation
        remains based on a transparent normalized finding dictionary
        with explicit handling of negation and uncertainty.
        """
    )

    st.markdown(
        "### Why pretrained models?"
    )

    st.markdown(
        """
        Training a clinically meaningful radiology model from
        scratch would require a substantially larger dataset,
        expert annotations, compute resources and formal validation.

        This project therefore focuses on system design:
        integrating pretrained models, clinical text processing,
        structured representations and multimodal comparison into
        one inspectable application.
        """
    )

    st.markdown(
        "### Model interpretation"
    )

    st.markdown(
        """
        A model score represents the output of a particular
        pretrained classifier under its training distribution.
        It should not be interpreted as a statement that a disease
        is present with that exact probability.

        The application therefore deliberately avoids converting
        model scores into diagnostic statements.
        """
    )


# ============================================================
# PAGE: LIMITATIONS
# ============================================================

elif page == "Limitations":

    st.markdown(
        '<div class="app-kicker">Responsible research</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="app-title">Limitations</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="app-subtitle">
            Understanding what the system cannot establish is part
            of the design of a responsible medical AI prototype.
        </div>
        """,
        unsafe_allow_html=True
    )

    limitations = [
        (
            "Not clinically validated",
            "The application has not undergone clinical validation, regulatory review or prospective testing."
        ),
        (
            "Model scores are not diagnoses",
            "The vision model produces research-model scores. These should not be interpreted as confirmed clinical findings."
        ),
        (
            "Report agreement is not ground truth",
            "A model agreeing with a radiology report does not independently establish model correctness."
        ),
        (
            "Dataset shift",
            "Performance can change across hospitals, scanners, populations, acquisition protocols and disease prevalence."
        ),
        (
            "Text extraction limitations",
            "Simple terminology and context rules cannot represent every form of clinical language."
        ),
        (
            "No treatment recommendations",
            "The system intentionally does not recommend medication, treatment or clinical management."
        ),
        (
            "No persistent patient database",
            "The prototype stores case history only within the current session."
        ),
        (
            "Public deployment",
            "Users should not upload identifiable patient information to a public research deployment."
        ),
    ]

    for title, description in limitations:

        st.markdown(
            f"""
            <div class="surface">

                <h3 style="margin-top:0;">
                    {title}
                </h3>

                <div class="small-note">
                    {description}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# PAGE: ABOUT
# ============================================================

elif page == "About":

    st.markdown(
        '<div class="app-kicker">Project overview</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="app-title">About the Project</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="app-subtitle">
            An independent exploration of multimodal machine
            learning for medical information interpretation.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        "### Research motivation"
    )

    st.markdown(
        """
        Radiology examinations generate more than one source of
        information. The image contains visual evidence, while the
        accompanying report contains a clinician's textual
        interpretation.

        This project explores whether these two modalities can be
        represented independently and then compared in a transparent
        computational pipeline.
        """
    )

    st.markdown(
        "### Research question"
    )

    st.markdown(
        """
        Can a lightweight multimodal system combine pretrained
        chest-radiograph analysis with structured clinical-text
        processing to make relationships between image-model
        outputs and report findings easier to inspect?
        """
    )

    st.markdown(
        "### What makes the prototype multimodal?"
    )

    st.markdown(
        """
        The system does not simply pass a radiology report to an
        LLM or classify an image in isolation.

        It creates two structured representations:

        **Visual representation**

        Chest X-ray → pretrained vision model → radiographic scores

        **Language representation**

        Radiology report → terminology normalization →
        negation/uncertainty detection → structured findings

        These representations are then compared at the finding
        level.
        """
    )

    st.markdown(
        "### Project scope"
    )

    scope_df = pd.DataFrame(
        [
            ["Computer Vision", "Chest X-ray classification"],
            ["Natural Language Processing", "Radiology finding extraction"],
            ["Multimodal AI", "Image-report comparison"],
            ["Data Engineering", "PDF/TXT parsing and normalization"],
            ["Visualization", "Structured findings and model profiles"],
            ["Software Engineering", "Interactive deployed application"],
            ["Responsible AI", "Explicit limitations and uncertainty"],
        ],
        columns=[
            "Area",
            "Contribution",
        ],
    )

    st.dataframe(
        scope_df,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(
        """
        <div class="disclaimer">
        <strong>Project status.</strong>
        This is an independent educational and research prototype.
        It is not a medical device, diagnostic system or substitute
        for professional radiological interpretation.
        </div>
        """,
        unsafe_allow_html=True
    )
