import io
import json
import os
import re
import uuid

import pandas as pd
import streamlit as st
from PIL import Image


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Multimodal Medical Report Visualizer",
    page_icon="🩻",
    layout="wide",
)


# ============================================================
# CONFIGURATION
# ============================================================

VISION_MODEL = os.getenv(
    "VISION_MODEL",
    "itsomk/chexpert-densenet121",
)

NLP_MODEL = os.getenv(
    "NLP_MODEL",
    "d4data/biomedical-ner-all",
)


# ============================================================
# CHEXPERT LABELS
# ============================================================

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


VISION_TO_CANONICAL = {
    "no finding": "no_finding",
    "enlarged cardiomediastinum": "enlarged_cardiomediastinum",
    "cardiomegaly": "cardiomegaly",
    "lung opacity": "lung_opacity",
    "lung lesion": "lung_lesion",
    "edema": "edema",
    "consolidation": "consolidation",
    "pneumonia": "pneumonia",
    "atelectasis": "atelectasis",
    "pneumothorax": "pneumothorax",
    "pleural effusion": "pleural_effusion",
    "pleural other": "pleural_other",
    "fracture": "fracture",
    "support devices": "support_devices",
}


# ============================================================
# RADIOLOGY VOCABULARY
# ============================================================

REPORT_TERMS = {

    "lung opacity": (
        "lung_opacity",
        "An area of the lung looks different or denser "
        "than surrounding lung tissue.",
    ),

    "airspace opacity": (
        "lung_opacity",
        "An area of the lung appears denser "
        "than surrounding lung tissue.",
    ),

    "opacity": (
        "lung_opacity",
        "An area that looks different from surrounding "
        "lung tissue.",
    ),

    "cardiomegaly": (
        "cardiomegaly",
        "The heart appears larger than usual.",
    ),

    "pleural effusion": (
        "pleural_effusion",
        "Extra fluid is described around the lung.",
    ),

    "pneumothorax": (
        "pneumothorax",
        "Air is described in the space around the lung.",
    ),

    "consolidation": (
        "consolidation",
        "An area of lung is described as abnormally dense.",
    ),

    "pneumonia": (
        "pneumonia",
        "The report uses the term pneumonia.",
    ),

    "atelectasis": (
        "atelectasis",
        "Part of the lung is described as partially "
        "collapsed or under-expanded.",
    ),

    "edema": (
        "edema",
        "The report describes fluid-related changes.",
    ),

    "lung lesion": (
        "lung_lesion",
        "A focal abnormality in the lung is described.",
    ),

    "pleural other": (
        "pleural_other",
        "Another abnormality involving the pleural "
        "space is described.",
    ),

    "fracture": (
        "fracture",
        "A fracture is described.",
    ),

    "support device": (
        "support_devices",
        "A medical support device is described.",
    ),
}


NEGATION_PATTERNS = [
    r"\bno\b",
    r"\bwithout\b",
    r"\bnegative for\b",
    r"\babsence of\b",
    r"\bfree of\b",
    r"\bnot seen\b",
    r"\bnot identified\b",
    r"\bno evidence of\b",
]


UNCERTAINTY_PATTERNS = [
    r"\bpossible\b",
    r"\bpossibly\b",
    r"\bmay represent\b",
    r"\bcannot exclude\b",
    r"\bsuspicious for\b",
    r"\bquestion of\b",
    r"\bconcerning for\b",
]


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
<style>

.block-container {
    max-width: 1450px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.hero {
    padding: 1.7rem 1.8rem;
    border: 1px solid rgba(128,128,128,.22);
    border-radius: 20px;
    background:
        linear-gradient(
            135deg,
            rgba(37,99,235,.10),
            rgba(14,165,233,.05)
        );
}

.muted {
    color: #6b7280;
}

.disclaimer {
    padding: 1rem 1.1rem;
    border-radius: 14px;
    background: rgba(245,158,11,.10);
    border: 1px solid rgba(245,158,11,.25);
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# VISION MODEL
# ============================================================

@st.cache_resource(show_spinner=False)
def load_vision_model():

    """
    Loads the actual CheXpert DenseNet-121 checkpoint.

    The Hugging Face repository contains a PyTorch/
    torchvision DenseNet-121 checkpoint stored as
    pytorch_model.safetensors.

    Therefore we load it directly instead of using
    transformers.pipeline().
    """

    import torch

    from torchvision import models, transforms

    from safetensors.torch import load_file

    from huggingface_hub import hf_hub_download


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
                num_labels,
            )

        def forward(self, x):

            return self.densenet(x)


    # --------------------------------------------------------
    # Download checkpoint from Hugging Face
    # --------------------------------------------------------

    checkpoint_path = hf_hub_download(
        repo_id=VISION_MODEL,
        filename="pytorch_model.safetensors",
    )


    # --------------------------------------------------------
    # Create model architecture
    # --------------------------------------------------------

    model = DenseNet121CheXpert(
        num_labels=14
    )


    # --------------------------------------------------------
    # Load safetensors checkpoint
    # --------------------------------------------------------

    state_dict = load_file(
        checkpoint_path
    )


    missing_keys, unexpected_keys = (
        model.load_state_dict(
            state_dict,
            strict=False,
        )
    )


    # --------------------------------------------------------
    # Safety check
    # --------------------------------------------------------

    if len(state_dict) < 100:

        raise RuntimeError(
            "The CheXpert checkpoint did not load "
            "correctly. The downloaded checkpoint "
            "contains unexpectedly few parameters."
        )


    model.eval()


    # --------------------------------------------------------
    # Image preprocessing
    # --------------------------------------------------------

    preprocess = transforms.Compose(
        [

            transforms.Resize(
                (224, 224)
            ),

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


    return (
        model,
        preprocess,
        torch,
    )


# ============================================================
# X-RAY INFERENCE
# ============================================================

def analyze_xray(
    image_bytes
):

    """
    Runs real multi-label inference on
    the uploaded chest X-ray.
    """

    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")


    model, preprocess, torch = (
        load_vision_model()
    )


    tensor = preprocess(
        image
    ).unsqueeze(0)


    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    with torch.inference_mode():

        logits = model(
            tensor
        )

        probabilities = (
            torch.sigmoid(
                logits
            )
            .squeeze(0)
            .tolist()
        )


    findings = []


    for label, probability in zip(
        CHEXPERT_LABELS,
        probabilities,
    ):

        canonical_name = (
            VISION_TO_CANONICAL[
                label.lower()
            ]
        )


        probability = float(
            probability
        )


        findings.append(
            {

                "name":
                    canonical_name,

                "display_name":
                    label,

                "probability":
                    probability,

                "status":
                    (
                        "elevated"
                        if probability >= 0.5
                        else "low"
                    ),

                "source":
                    "Hugging Face CheXpert DenseNet-121",

                "explanation":
                    (
                        "Pretrained chest-X-ray "
                        "model output. This probability "
                        "is not a clinical diagnosis."
                    ),
            }
        )


    findings.sort(
        key=lambda item:
            item["probability"],
        reverse=True,
    )


    return findings


# ============================================================
# PDF EXTRACTION
# ============================================================

def extract_pdf_text(
    file_bytes
):

    try:

        import fitz


        document = fitz.open(
            stream=file_bytes,
            filetype="pdf",
        )


        pages = []


        for page in document:

            pages.append(
                page.get_text()
            )


        return "\n".join(
            pages
        ).strip()


    except Exception:

        return ""


# ============================================================
# REPORT FILE READING
# ============================================================

def read_report(
    uploaded_file
):

    if uploaded_file is None:

        return ""


    file_bytes = (
        uploaded_file.getvalue()
    )


    filename = (
        uploaded_file.name.lower()
    )


    if filename.endswith(
        ".pdf"
    ):

        return extract_pdf_text(
            file_bytes
        )


    return file_bytes.decode(
        "utf-8",
        errors="ignore",
    )


# ============================================================
# REPORT NLP
# ============================================================

def extract_report_findings(
    report_text
):

    text = " ".join(
        report_text.split()
    )

    lower_text = text.lower()


    findings = []


    for term, (
        canonical_name,
        plain_language,
    ) in REPORT_TERMS.items():


        matches = list(
            re.finditer(
                re.escape(term),
                lower_text,
            )
        )


        if not matches:

            continue


        match = matches[0]


        start = max(
            0,
            match.start() - 100,
        )


        end = min(
            len(lower_text),
            match.end() + 140,
        )


        context = (
            lower_text[
                start:end
            ]
        )


        # ----------------------------------------------------
        # Negation detection
        # ----------------------------------------------------

        negated = any(
            re.search(
                pattern,
                context,
            )
            for pattern
            in NEGATION_PATTERNS
        )


        # ----------------------------------------------------
        # Uncertainty detection
        # ----------------------------------------------------

        uncertain = any(
            re.search(
                pattern,
                context,
            )
            for pattern
            in UNCERTAINTY_PATTERNS
        )


        if negated:

            status = "absent"

        elif uncertain:

            status = "uncertain"

        else:

            status = "present"


        evidence = (
            text[
                start:end
            ].strip()
        )


        # ----------------------------------------------------
        # Basic anatomical location
        # ----------------------------------------------------

        location = None


        location_patterns = [

            (
                "right lower",
                "right lower lung",
            ),

            (
                "left lower",
                "left lower lung",
            ),

            (
                "right upper",
                "right upper lung",
            ),

            (
                "left upper",
                "left upper lung",
            ),

            (
                "right middle",
                "right middle lung",
            ),

            (
                "left mid",
                "left mid lung",
            ),
        ]


        for phrase, readable in (
            location_patterns
        ):

            if phrase in context:

                location = readable

                break


        findings.append(
            {

                "name":
                    canonical_name,

                "display_name":
                    term.title(),

                "status":
                    status,

                "location":
                    location,

                "evidence":
                    evidence,

                "plain_language":
                    plain_language,
            }
        )


    # --------------------------------------------------------
    # Remove duplicate canonical findings
    # --------------------------------------------------------

    unique_findings = {}


    for finding in findings:

        if finding["name"] not in unique_findings:

            unique_findings[
                finding["name"]
            ] = finding


    findings = list(
        unique_findings.values()
    )


    # --------------------------------------------------------
    # Biomedical NER fallback
    # --------------------------------------------------------

    if not findings:

        try:

            from transformers import pipeline


            ner_pipeline = pipeline(
                "token-classification",
                model=NLP_MODEL,
                aggregation_strategy="simple",
            )


            entities = ner_pipeline(
                text[:4000]
            )


            for entity in entities[:20]:

                word = (
                    entity
                    .get("word", "")
                    .strip()
                )


                if not word:

                    continue


                findings.append(
                    {

                        "name":
                            word.lower()
                            .replace(
                                " ",
                                "_",
                            ),

                        "display_name":
                            word,

                        "status":
                            "uncertain",

                        "location":
                            None,

                        "evidence":
                            word,

                        "plain_language":
                            (
                                "A biomedical term "
                                "identified in the report. "
                                "Its clinical meaning "
                                "requires context."
                            ),
                    }
                )


        except Exception:

            pass


    # --------------------------------------------------------
    # Nothing identified
    # --------------------------------------------------------

    if not findings:

        findings.append(
            {

                "name":
                    "no_normalized_findings",

                "display_name":
                    "No normalized findings",

                "status":
                    "uncertain",

                "location":
                    None,

                "evidence":
                    "",

                "plain_language":
                    (
                        "The prototype did not identify "
                        "one of its supported radiology "
                        "finding patterns."
                    ),
            }
        )


    return findings


# ============================================================
# MULTIMODAL FUSION
# ============================================================

def compare_findings(
    vision_findings,
    report_findings,
):

    report_map = {

        finding["name"]:
            finding

        for finding
        in report_findings

        if finding["name"]
        != "no_normalized_findings"
    }


    comparisons = []


    for vision in vision_findings:

        name = vision["name"]


        if name == "no_finding":

            continue


        probability = float(
            vision["probability"]
        )


        image_positive = (
            probability >= 0.5
        )


        if name not in report_map:

            relationship = (
                "not_mentioned"
            )

            report_status = (
                "not mentioned"
            )


        else:

            report_status = (
                report_map[name]["status"]
            )


            if report_status == "uncertain":

                relationship = "uncertain"


            elif (
                image_positive
                and
                report_status == "present"
            ):

                relationship = "consistent"


            elif (
                not image_positive
                and
                report_status == "absent"
            ):

                relationship = "consistent"


            else:

                relationship = (
                    "potential_difference"
                )


        comparisons.append(
            {

                "finding":
                    name,

                "display_name":
                    vision["display_name"],

                "image_probability":
                    probability,

                "image":
                    f"{probability:.0%}",

                "report":
                    report_status,

                "relationship":
                    relationship,
            }
        )


    return comparisons


# ============================================================
# PLAIN LANGUAGE EXPLANATION
# ============================================================

def deterministic_summary(
    report_findings,
    comparisons,
):

    present = [

        finding["display_name"]

        for finding
        in report_findings

        if finding["status"]
        == "present"
    ]


    absent = [

        finding["display_name"]

        for finding
        in report_findings

        if finding["status"]
        == "absent"
    ]


    consistent = [

        finding["display_name"]

        for finding
        in comparisons

        if finding["relationship"]
        == "consistent"
    ]


    parts = []


    if present:

        parts.append(
            "The written report describes: "
            + ", ".join(present)
            + "."
        )


    if absent:

        parts.append(
            "The report specifically describes "
            "as absent: "
            + ", ".join(absent)
            + "."
        )


    if consistent:

        parts.append(
            "The image-model outputs are directionally "
            "consistent with the written report for: "
            + ", ".join(consistent)
            + "."
        )


    if not parts:

        parts.append(
            "The prototype could not generate "
            "a detailed plain-language summary "
            "from the supplied information."
        )


    parts.append(
        "This is an AI-assisted educational explanation. "
        "It is not a medical diagnosis and should not "
        "be used for treatment or clinical decision-making."
    )


    return " ".join(
        parts
    )


# ============================================================
# OPTIONAL LLM EXPLANATION
# ============================================================

def generate_explanation(
    report_findings,
    comparisons,
):

    api_key = os.getenv(
        "LLM_API_KEY"
    )


    provider = os.getenv(
        "LLM_PROVIDER",
        "",
    ).lower()


    # --------------------------------------------------------
    # Free default mode
    # --------------------------------------------------------

    if (
        not api_key
        or provider != "openai"
    ):

        return deterministic_summary(
            report_findings,
            comparisons,
        )


    # --------------------------------------------------------
    # Optional OpenAI API
    # --------------------------------------------------------

    try:

        from openai import OpenAI


        client = OpenAI(
            api_key=api_key
        )


        structured_data = {

            "report_findings":
                report_findings,

            "image_report_comparisons":
                comparisons,
        }


        response = (
            client
            .chat
            .completions
            .create(

                model=os.getenv(
                    "LLM_MODEL",
                    "gpt-5-mini",
                ),

                messages=[

                    {
                        "role":
                            "system",

                        "content":
                            (
                                "Explain radiology "
                                "information for a general "
                                "audience. The written "
                                "clinician report is the "
                                "primary source. AI image "
                                "outputs are predictions, "
                                "not diagnoses. Do not "
                                "invent findings, provide "
                                "treatment advice, or "
                                "claim clinical accuracy."
                            ),
                    },

                    {
                        "role":
                            "user",

                        "content":
                            json.dumps(
                                structured_data,
                                ensure_ascii=False,
                            ),
                    },
                ],

                temperature=0.1,
            )
        )


        return (
            response
            .choices[0]
            .message
            .content
            .strip()
        )


    except Exception:

        return deterministic_summary(
            report_findings,
            comparisons,
        )


# ============================================================
# DEMO DATA
# ============================================================

def demo_result():

    study_id = (
        "DEMO-"
        + uuid.uuid4()
        .hex[:6]
        .upper()
    )


    vision = [

        {
            "name":
                "lung_opacity",

            "display_name":
                "Lung Opacity",

            "probability":
                0.78,

            "status":
                "elevated",

            "source":
                "Illustrative demo",

            "explanation":
                "Illustrative demo probability only.",
        },

        {
            "name":
                "pleural_effusion",

            "display_name":
                "Pleural Effusion",

            "probability":
                0.10,

            "status":
                "low",

            "source":
                "Illustrative demo",

            "explanation":
                "Illustrative demo probability only.",
        },

        {
            "name":
                "cardiomegaly",

            "display_name":
                "Cardiomegaly",

            "probability":
                0.21,

            "status":
                "low",

            "source":
                "Illustrative demo",

            "explanation":
                "Illustrative demo probability only.",
        },
    ]


    report = [

        {
            "name":
                "lung_opacity",

            "display_name":
                "Lung Opacity",

            "status":
                "present",

            "location":
                "right lower lung",

            "evidence":
                (
                    "Patchy right lower lobe "
                    "airspace opacity is described."
                ),

            "plain_language":
                REPORT_TERMS[
                    "lung opacity"
                ][1],
        },

        {
            "name":
                "pleural_effusion",

            "display_name":
                "Pleural Effusion",

            "status":
                "absent",

            "location":
                None,

            "evidence":
                "No pleural effusion is described.",

            "plain_language":
                REPORT_TERMS[
                    "pleural effusion"
                ][1],
        },
    ]


    comparisons = [

        {
            "finding":
                "lung_opacity",

            "display_name":
                "Lung Opacity",

            "image_probability":
                0.78,

            "image":
                "78%",

            "report":
                "present",

            "relationship":
                "consistent",
        },

        {
            "finding":
                "pleural_effusion",

            "display_name":
                "Pleural Effusion",

            "image_probability":
                0.10,

            "image":
                "10%",

            "report":
                "absent",

            "relationship":
                "consistent",
        },

        {
            "finding":
                "cardiomegaly",

            "display_name":
                "Cardiomegaly",

            "image_probability":
                0.21,

            "image":
                "21%",

            "report":
                "not mentioned",

            "relationship":
                "not_mentioned",
        },
    ]


    return {

        "study_id":
            study_id,

        "demo":
            True,

        "summary":
            (
                "This illustrative case contains "
                "a reported lung opacity and no "
                "reported pleural effusion. The "
                "example image-model outputs are "
                "directionally consistent with "
                "those example statements. These "
                "values are synthetic and are not "
                "a diagnosis."
            ),

        "vision_findings":
            vision,

        "report_findings":
            report,

        "comparisons":
            comparisons,

        "medical_terms":
            [

                {
                    "term":
                        "Lung opacity",

                    "plain_language":
                        REPORT_TERMS[
                            "lung opacity"
                        ][1],
                },

                {
                    "term":
                        "Pleural effusion",

                    "plain_language":
                        REPORT_TERMS[
                            "pleural effusion"
                        ][1],
                },

                {
                    "term":
                        "Cardiomegaly",

                    "plain_language":
                        REPORT_TERMS[
                            "cardiomegaly"
                        ][1],
                },
            ],

        "limitations":
            [

                "Demo values are illustrative.",

                "Model probabilities are not clinical certainty.",

                "This prototype is not clinically validated.",
            ],

        "models":
            {

                "vision":
                    VISION_MODEL,

                "nlp":
                    NLP_MODEL,

                "explanation":
                    "Deterministic / optional LLM API",
            },
    }


# ============================================================
# COMPLETE ANALYSIS PIPELINE
# ============================================================

def analyze_case(
    image_bytes,
    report_text,
):

    study_id = (
        "STUDY-"
        + uuid.uuid4()
        .hex[:8]
        .upper()
    )


    # --------------------------------------------------------
    # Computer vision
    # --------------------------------------------------------

    vision_findings = analyze_xray(
        image_bytes
    )


    # --------------------------------------------------------
    # Report NLP
    # --------------------------------------------------------

    report_findings = (
        extract_report_findings(
            report_text
        )
    )


    # --------------------------------------------------------
    # Multimodal fusion
    # --------------------------------------------------------

    comparisons = (
        compare_findings(
            vision_findings,
            report_findings,
        )
    )


    # --------------------------------------------------------
    # Explanation
    # --------------------------------------------------------

    summary = (
        generate_explanation(
            report_findings,
            comparisons,
        )
    )


    # --------------------------------------------------------
    # Medical terminology
    # --------------------------------------------------------

    medical_terms = []

    seen = set()


    for finding in report_findings:

        if finding["name"] in seen:

            continue


        seen.add(
            finding["name"]
        )


        medical_terms.append(
            {

                "term":
                    finding["display_name"],

                "plain_language":
                    finding["plain_language"],
            }
        )


    return {

        "study_id":
            study_id,

        "demo":
            False,

        "summary":
            summary,

        "vision_findings":
            vision_findings,

        "report_findings":
            report_findings,

        "comparisons":
            comparisons,

        "medical_terms":
            medical_terms,

        "limitations":
            [

                "AI observations are not diagnoses.",

                "The written radiology report "
                "is treated as the primary clinical source.",

                "The 0.5 threshold is an engineering "
                "display threshold, not a clinical threshold.",

                "This prototype has not undergone "
                "clinical validation.",

            ],

        "models":
            {

                "vision":
                    VISION_MODEL,

                "nlp":
                    NLP_MODEL,

                "explanation":
                    (
                        os.getenv(
                            "LLM_MODEL",
                            "Deterministic / optional LLM API",
                        )
                    ),
            },
    }


# ============================================================
# JSON EXPORT
# ============================================================

def export_json(
    result
):

    return json.dumps(
        result,
        indent=2,
        ensure_ascii=False,
    )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "🩻 Medical Visualizer"
)


page = st.sidebar.radio(
    "Navigate",
    [

        "Analyze",

        "History",

        "Evaluation",

        "Models & Methodology",

        "About",

    ],
)


if "history" not in st.session_state:

    st.session_state.history = []


# ============================================================
# ANALYZE PAGE
# ============================================================

if page == "Analyze":

    st.markdown(
        """
<div class="hero">

<h1>
Multimodal Medical Report Visualizer
</h1>

<p>
Explore how a pretrained chest-X-ray computer-vision
model and biomedical report analysis can be combined
to make radiology information easier to inspect.
</p>

</div>
""",
        unsafe_allow_html=True,
    )


    st.write("")


    left, right = st.columns(
        2
    )


    # ========================================================
    # X-RAY
    # ========================================================

    with left:

        st.subheader(
            "1. Chest X-ray"
        )


        xray = st.file_uploader(
            "Upload PNG / JPG / JPEG",

            type=[
                "png",
                "jpg",
                "jpeg",
            ],

            key="xray_upload",
        )


        if xray:

            st.image(
                xray,

                caption="Uploaded X-ray",

                use_container_width=True,
            )


    # ========================================================
    # REPORT
    # ========================================================

    with right:

        st.subheader(
            "2. Radiology report"
        )


        report_file = st.file_uploader(
            "Upload PDF / TXT",

            type=[
                "pdf",
                "txt",
            ],

            key="report_upload",
        )


        uploaded_report = (
            read_report(
                report_file
            )
        )


        report_text = st.text_area(
            "Paste or edit the report",

            value=uploaded_report,

            height=240,

            placeholder=(
                "Example: No pleural effusion. "
                "Patchy right lower lobe opacity..."
            ),
        )


    st.divider()


    col1, col2, col3 = st.columns(
        [1, 1, 2]
    )


    with col1:

        demo_clicked = st.button(
            "🎬 Load Demo Case",

            use_container_width=True,
        )


    with col2:

        analyze_clicked = st.button(
            "🔬 Analyze Study",

            type="primary",

            use_container_width=True,
        )


    with col3:

        st.caption(
            "Educational/research prototype — "
            "not a diagnostic system."
        )


    # ========================================================
    # DEMO
    # ========================================================

    if demo_clicked:

        result = demo_result()

        st.session_state.result = result


    # ========================================================
    # REAL ANALYSIS
    # ========================================================

    if analyze_clicked:

        if not xray:

            st.error(
                "Please upload a chest X-ray."
            )


        elif not report_text.strip():

            st.error(
                "Please upload or paste "
                "a radiology report."
            )


        else:

            with st.status(
                "Running multimodal analysis...",
                expanded=True,
            ) as status:

                try:

                    st.write(
                        "✓ Image loaded"
                    )


                    st.write(
                        "✓ Report loaded"
                    )


                    image_bytes = (
                        xray.getvalue()
                    )


                    result = analyze_case(
                        image_bytes,
                        report_text,
                    )


                    st.write(
                        "✓ DenseNet-121 vision inference"
                    )


                    st.write(
                        "✓ Report parsing / biomedical NLP"
                    )


                    st.write(
                        "✓ Finding normalization"
                    )


                    st.write(
                        "✓ Image ↔ report comparison"
                    )


                    st.write(
                        "✓ Explanation generation"
                    )


                    status.update(
                        label="Analysis complete",

                        state="complete",
                    )


                    st.session_state.result = (
                        result
                    )


                    st.session_state.history.insert(
                        0,
                        result,
                    )


                except Exception as exc:

                    status.update(
                        label="Analysis failed",

                        state="error",
                    )


                    st.error(
                        "The live model pipeline failed."
                    )


                    # This is deliberately visible so
                    # deployment/model problems can be
                    # diagnosed instead of hidden.

                    st.exception(
                        exc
                    )


    # ========================================================
    # RESULTS
    # ========================================================

    result = (
        st.session_state.get(
            "result"
        )
    )


    if result:

        st.divider()


        st.header(
            f"Study {result['study_id']}"
        )


        if result.get("demo"):

            st.info(
                "Demo mode: all values are illustrative."
            )


        # ----------------------------------------------------
        # SUMMARY
        # ----------------------------------------------------

        st.subheader(
            "Plain-language summary"
        )


        st.info(
            result["summary"]
        )


        # ----------------------------------------------------
        # AI FINDINGS
        # ----------------------------------------------------

        st.subheader(
            "AI visual findings"
        )


        vision = (
            result[
                "vision_findings"
            ]
        )


        # Display top 9 findings

        top_findings = vision[:9]


        for start in range(
            0,
            len(top_findings),
            3,
        ):

            cols = st.columns(
                3
            )


            for offset, col in enumerate(
                cols
            ):

                index = (
                    start
                    + offset
                )


                if (
                    index
                    >= len(top_findings)
                ):

                    continue


                finding = (
                    top_findings[
                        index
                    ]
                )


                with col:

                    with st.container(
                        border=True
                    ):

                        st.markdown(
                            "### "
                            + finding[
                                "display_name"
                            ]
                        )


                        st.metric(
                            "Model probability",

                            f"{finding['probability']:.1%}",
                        )


                        st.progress(
                            min(
                                1.0,

                                max(
                                    0.0,

                                    finding[
                                        "probability"
                                    ],
                                ),
                            )
                        )


                        st.caption(
                            finding[
                                "explanation"
                            ]
                        )


        # ----------------------------------------------------
        # REPORT FINDINGS
        # ----------------------------------------------------

        st.subheader(
            "Radiology report findings"
        )


        for finding in (
            result[
                "report_findings"
            ]
        ):

            with st.container(
                border=True
            ):

                st.markdown(
                    "### "
                    + finding[
                        "display_name"
                    ]
                )


                st.write(
                    "**Status:** "
                    + finding[
                        "status"
                    ]
                    .replace(
                        "_",
                        " ",
                    )
                    .title()
                )


                if finding.get(
                    "location"
                ):

                    st.write(
                        "**Location:** "
                        + finding[
                            "location"
                        ]
                    )


                if finding.get(
                    "evidence"
                ):

                    st.caption(
                        "Report evidence: "
                        + finding[
                            "evidence"
                        ]
                    )


        # ----------------------------------------------------
        # COMPARISON
        # ----------------------------------------------------

        st.subheader(
            "Image ↔ report comparison"
        )


        comparison_rows = []


        for item in (
            result[
                "comparisons"
            ]
        ):

            comparison_rows.append(
                {

                    "Finding":
                        item[
                            "display_name"
                        ],

                    "Image model":
                        item[
                            "image"
                        ],

                    "Report":
                        item[
                            "report"
                        ].title(),

                    "Relationship":
                        item[
                            "relationship"
                        ]
                        .replace(
                            "_",
                            " ",
                        )
                        .title(),
                }
            )


        if comparison_rows:

            st.dataframe(
                pd.DataFrame(
                    comparison_rows
                ),

                use_container_width=True,

                hide_index=True,
            )


        st.caption(
            "Consistent means the prototype outputs "
            "point in the same direction for that "
            "finding. It does not establish clinical "
            "accuracy and does not imply that either "
            "source is correct."
        )


        # ----------------------------------------------------
        # MEDICAL TERMS
        # ----------------------------------------------------

        st.subheader(
            "Medical terminology"
        )


        for term in (
            result[
                "medical_terms"
            ]
        ):

            with st.expander(
                term["term"]
            ):

                st.write(
                    term[
                        "plain_language"
                    ]
                )


        # ----------------------------------------------------
        # CHART
        # ----------------------------------------------------

        st.subheader(
            "Model probability profile"
        )


        chart_data = pd.DataFrame(
            [

                {

                    "Finding":
                        item[
                            "display_name"
                        ],

                    "Probability":
                        item[
                            "probability"
                        ],

                }

                for item
                in vision

            ]
        )


        if not chart_data.empty:

            st.bar_chart(
                chart_data.set_index(
                    "Finding"
                )
            )


        # ----------------------------------------------------
        # EXPORT
        # ----------------------------------------------------

        st.subheader(
            "Export"
        )


        st.download_button(
            "Download JSON analysis",

            data=export_json(
                result
            ),

            file_name=(
                f"{result['study_id']}.json"
            ),

            mime="application/json",
        )


        # ----------------------------------------------------
        # DISCLAIMER
        # ----------------------------------------------------

        st.markdown(
            """
<div class="disclaimer">

<b>Responsible use:</b>

AI observations are model outputs,
not diagnoses.

The clinician/radiology report is
the primary clinical source.

Do not use this application for
treatment or medical decision-making.

Do not upload identifiable
patient information.

</div>
""",
            unsafe_allow_html=True,
        )


# ============================================================
# HISTORY PAGE
# ============================================================

elif page == "History":

    st.title(
        "Study History"
    )


    if not st.session_state.history:

        st.info(
            "No analyses in this browser "
            "session yet."
        )


    else:

        for result in (
            st.session_state.history
        ):

            with st.expander(
                (
                    f"{result['study_id']} — "
                    f"{result['summary'][:100]}"
                )
            ):

                st.write(
                    result[
                        "summary"
                    ]
                )


                st.download_button(
                    "Download JSON",

                    data=export_json(
                        result
                    ),

                    file_name=(
                        f"{result['study_id']}.json"
                    ),

                    mime="application/json",

                    key=(
                        "history_"
                        + result[
                            "study_id"
                        ]
                    ),
                )


# ============================================================
# EVALUATION PAGE
# ============================================================

elif page == "Evaluation":

    st.title(
        "Evaluation"
    )


    st.write(
        "This page deliberately avoids "
        "claiming clinical accuracy from "
        "agreement with a radiology report."
    )


    st.markdown(
        """
### Proper evaluation design

For a real research evaluation:

1. Use an appropriately licensed,
   de-identified dataset.

2. Define the label ontology before testing.

3. Keep patient-level train/validation/test
   separation.

4. Use an independent reference standard.

5. Report AUROC, sensitivity, specificity,
   and confidence intervals where appropriate.

6. Evaluate uncertain and missing labels.

7. Perform subgroup analysis where the
   dataset supports it.

The built-in demo is illustrative and
is not a clinical benchmark.
"""
    )


# ============================================================
# MODELS & METHODOLOGY
# ============================================================

elif page == "Models & Methodology":

    st.title(
        "Models & Methodology"
    )


    st.markdown(
        f"""
### Computer vision

**Model:**
`{VISION_MODEL}`

The application uses a pretrained
DenseNet-121 chest-X-ray multi-label
classifier hosted on Hugging Face.

The model predicts 14 CheXpert-style
labels.

The application does not train the
model.

### Biomedical NLP

**Model:**
`{NLP_MODEL}`

The report pipeline first applies
radiology-specific normalization and
local-context negation handling.

The biomedical NER model is used only
as a fallback when the supported
radiology vocabulary does not identify
a finding.

### Multimodal fusion

```text
Chest X-ray
      |
      v
DenseNet-121
      |
      v
Image findings
      |
      |
      +----------------------+
                             |
                             v
                      Normalization
                             ^
                             |
                             |
Radiology report             |
      |                      |
      v                      |
Report NLP ------------------+
      |
      v
Report findings
      |
      v
Image ↔ Report comparison
      |
      v
Plain-language explanation
