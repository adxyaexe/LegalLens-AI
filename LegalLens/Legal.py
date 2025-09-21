"""
LegalLens AI - Streamlit prototype (GEMINI VERSION)

Features:
- Upload PDF/DOCX or paste contract text
- Clause splitting (heuristic)
- Plain-English clause summarization via Google Gemini
- Risk Radar: regex/keyword-based detector
- Persona-specific summaries (Tenant / Employee / Freelancer)
- Interactive Q&A on the uploaded contract

How to run:
1. Create a Python venv and install requirements:
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   pip install -r requirements.txt

requirements.txt (create this file):
streamlit
google-generativeai
pdfplumber
python-docx
tqdm
regex
python-dotenv

2. Create a file named .env in the same directory and add your Google API key:
   GOOGLE_API_KEY="AIzaSy..."

3. Run the app:
   streamlit run Legal.py
"""

import os
import re
import io
import json
import streamlit as st
import google.generativeai as genai
from typing import List, Dict, Tuple
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Optional: pdfplumber and python-docx for extraction
try:
    import pdfplumber
except ImportError:
    pdfplumber = None
try:
    import docx
except ImportError:
    docx = None

# ---------- CONFIG ----------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

DEFAULT_MODEL = "gemini-1.5-flash"

# Risk keywords and regex - tune for your domain
RISK_KEYWORDS = {
    'auto_renew': [r"auto-?renew", r"automatic renewal"],
    'termination_without_notice': [r"terminate.*without.*notice", r"immediate termination"],
    'one_sided_indemnity': [r"hold harmless", r"indemnif", r"liable for all"],
    'penalties': [r"penalt", r"late fee", r"liquidated damage"],
    'non_compete': [r"non-?compete", r"restrict.*compete"],
    'confidentiality': [r"confidential", r"non-disclos"],
    'governing_law': [r"govern(ed)? by", r"jurisdiction"],
}

RISK_LABELS = {
    'auto_renew': ('⚠️', 'Automatic renewal found — may trap users into continued obligations without clear opt-out.'),
    'termination_without_notice': ('❗', 'Possible termination without notice — risky for the weaker party.'),
    'one_sided_indemnity': ('❗', 'Indemnity or broad liability clauses that could be one-sided.'),
    'penalties': ('⚠️', 'Penalties or fees that may be excessive.'),
    'non_compete': ('❗', 'Non-compete clause detected — could restrict future opportunities.'),
    'confidentiality': ('ℹ️', 'Confidentiality clauses found — note data/IP implications.'),
    'governing_law': ('ℹ️', 'Governing law / jurisdiction clauses found — affects dispute resolution.'),
}

# ---------- PROMPTS ----------
SUMMARY_PROMPT = (
    "You are a helpful legal assistant.\n"
    "Simplify the following CLAUSE into plain English for a non-lawyer.\n"
    "Output ONLY a valid JSON object with keys: 'summary' (short plain-English summary), 'obligations' (list of strings), 'risks' (list of strings), 'suggested_actions' (list of strings).\n"
    "Be concise but accurate. Preserve the legal meaning. Do not add any commentary before or after the JSON.\n\n"
    "CLAUSE:\n\n"
)

PERSONA_PROMPT = (
    "You are a legal assistant writing for a {persona}. The persona: {persona_desc}.\n"
    "Given the clause below, summarize only what matters to this persona in 2-4 short bullet points.\n\n"
    "CLAUSE:\n\n"
)

QA_PROMPT = (
    "You are a legal assistant. Answer the user's question using ONLY the contract text provided. If the contract does not answer the question, say 'Not specified in the contract.'\n"
    "Cite the exact clause snippet (up to 30 words) that supports your answer.\n\n"
    "CONTRACT:\n\n"
)

# ---------- UTILITIES ----------

def extract_text_from_pdf_bytes(file_bytes: bytes) -> str:
    if not pdfplumber:
        return ""
    text = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)
    return "\n\n".join(text)


def extract_text_from_docx_bytes(file_bytes: bytes) -> str:
    if not docx:
        return ""
    document = docx.Document(io.BytesIO(file_bytes))
    paragraphs = [p.text for p in document.paragraphs if p.text.strip()]
    return "\n\n".join(paragraphs)


def heuristic_split_clauses(text: str) -> List[str]:
    t = text.replace('\r\n', '\n')
    parts = re.split(r"(?m)^(?:\d+\.|Section\s+\d+|\d+\))\s+", t)
    if len(parts) <= 1:
        parts = [p.strip() for p in t.split('\n\n') if len(p.strip()) > 30]
    clauses = [p.strip() for p in parts if p and len(p.strip()) > 30]
    merged, buffer = [], ''
    for c in clauses:
        if len(c) < 150 and buffer:
            buffer += '\n\n' + c
        else:
            if buffer: merged.append(buffer)
            buffer = c
    if buffer: merged.append(buffer)
    return merged


def detect_risks_in_clause(clause: str) -> List[Tuple[str,str,str]]:
    findings = []
    for key, patterns in RISK_KEYWORDS.items():
        for pat in patterns:
            if re.search(pat, clause, flags=re.IGNORECASE):
                label_icon, label_text = RISK_LABELS.get(key, ('⚠️','Risk'))
                snippet = extract_snippet(clause, pat)
                findings.append((key, label_icon, label_text + ' — snippet: "' + snippet + '"'))
                break
    return findings


def extract_snippet(text: str, pat: str, length: int = 60) -> str:
    m = re.search(pat, text, flags=re.IGNORECASE)
    if not m:
        return text[:length].strip()
    start, end = m.span()
    s = max(0, start - 30)
    e = min(len(text), end + 30)
    return text[s:e].strip().replace('\n',' ')


def heuristic_summary(text: str) -> str:
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    return ' '.join(sents[:2])


def clean_gemini_json_output(text: str) -> str:
    """Removes markdown backticks from Gemini's JSON output."""
    if text.strip().startswith("```json"):
        text = text.strip()[7:-3]
    return text.strip()

# ---------- GEMINI API CALLS ----------

def call_gemini_summarize(clause: str, model: str = DEFAULT_MODEL) -> dict:
    prompt = SUMMARY_PROMPT + clause
    try:
        gemini_model = genai.GenerativeModel(model)
        response = gemini_model.generate_content(prompt)
        cleaned_json = clean_gemini_json_output(response.text)
        return json.loads(cleaned_json)
    except Exception as e:
        st.error(f"Gemini API Error: {e}")
        return {"summary": heuristic_summary(clause), "obligations": [], "risks": [], "suggested_actions": ["API call failed."]}


def call_gemini_persona(clause: str, persona: str, persona_desc: str, model: str = DEFAULT_MODEL) -> str:
    prompt = PERSONA_PROMPT.format(persona=persona, persona_desc=persona_desc) + clause
    try:
        gemini_model = genai.GenerativeModel(model)
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Gemini API Error: {e}")
        return '- Key points: ' + heuristic_summary(clause)


def call_gemini_qa(contract_text: str, question: str, model: str = DEFAULT_MODEL) -> str:
    prompt = QA_PROMPT + contract_text + "\n\nQUESTION:\n\n" + question
    try:
        gemini_model = genai.GenerativeModel(model)
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Gemini API Error: {e}")
        return "Not available: Gemini API error."

# ---------- STREAMLIT UI ----------

st.set_page_config(page_title="LegalLens AI (Gemini)", layout="wide")
st.title("LegalLens AI — Powered by Gemini")

with st.sidebar:
    st.header("Prototype controls")
    model_choice = st.selectbox("Model", ["gemini-1.5-flash", "gemini-1.5-pro"], index=0)
    persona_choice = st.selectbox("Persona demo", ["Tenant", "Employee", "Freelancer"], index=0)
    st.markdown("---")
    st.write("Risk keywords (tune in code):")
    if st.button("Show risk keywords"):
        st.json(RISK_KEYWORDS)

uploaded_file = st.file_uploader("Upload a contract (PDF / DOCX) or paste text below", type=["pdf","docx","txt"] )
pasted = st.text_area("Or paste contract text (or clause)", height=200)

contract_text = ""
if uploaded_file is not None:
    if uploaded_file.name.lower().endswith('.pdf'):
        contract_text = extract_text_from_pdf_bytes(uploaded_file.read())
    elif uploaded_file.name.lower().endswith('.docx'):
        contract_text = extract_text_from_docx_bytes(uploaded_file.read())
    else:
        contract_text = uploaded_file.read().decode('utf-8')

if pasted and not contract_text:
    contract_text = pasted

if not contract_text:
    st.info("Upload or paste contract text to begin.")
    st.stop()

st.sidebar.markdown("---")
if GOOGLE_API_KEY:
    st.sidebar.success("Gemini API key detected — API calls enabled")
else:
    st.sidebar.warning("GOOGLE_API_KEY not set. Check your .env file.")

# Main processing
clauses = heuristic_split_clauses(contract_text)

st.subheader("Clauses detected")
col1, col2 = st.columns([1,2])
with col1:
    st.write(f"Detected {len(clauses)} clauses (heuristic split)")
    sel = st.selectbox("Select clause", list(range(len(clauses))), format_func=lambda i: "Clause {}: {}...".format(i + 1, clauses[i][:60].replace('\n', ' ')))
    clause = clauses[sel] if clauses else ""

with col2:
    st.markdown("**Original clause**")
    st.write(clause)

# Summarize selected clause
st.markdown("---")
st.subheader("Plain-English summary")
if st.button("Generate summary"):
    if not clause:
        st.warning("No clause selected.")
    else:
        with st.spinner("Calling Gemini..."):
            summ = call_gemini_summarize(clause, model=model_choice)
            st.markdown("**Summary:**")
            st.write(summ.get('summary', ''))
            st.markdown("**Obligations:**")
            for o in summ.get('obligations', []): st.write(f'- {o}')
            st.markdown("**Risks (from LLM):**")
            for r in summ.get('risks', []): st.write(f'- {r}')
            st.markdown("**Suggested actions:**")
            for a in summ.get('suggested_actions', []): st.write(f'- {a}')

# Risk radar heuristic
st.markdown("---")
st.subheader("Risk Radar (heuristic)")
if clause:
    findings = detect_risks_in_clause(clause)
    if findings:
        for key, icon, text in findings: st.write(f"{icon} **{key}** — {text}")
    else:
        st.success("No high-risk keywords detected in this clause.")

# Persona-specific summary
st.markdown("---")
st.subheader("Persona-specific summary")
persona_descs = {
    'Tenant': 'A residential tenant concerned about rent, deposit, termination, and penalties.',
    'Employee': 'An employee focused on compensation, non-compete, confidentiality, and termination.',
    'Freelancer': 'A gig worker interested in payments, IP ownership, and scope of work.'
}
if st.button("Generate persona summary"):
    if not clause:
        st.warning("No clause selected.")
    else:
        with st.spinner("Calling Gemini for persona..."):
            res = call_gemini_persona(clause, persona_choice, persona_descs[persona_choice], model=model_choice)
            st.write(res)

# Contract-level Q&A
st.markdown("---")
st.subheader("Ask questions about the contract")
question = st.text_input("Ask (e.g., 'What is the notice period for termination?')")
if st.button("Ask"):
    if not question.strip():
        st.warning("Type a question first.")
    else:
        with st.spinner("Finding answer with Gemini..."):
            ans = call_gemini_qa(contract_text, question, model=model_choice)
            st.write(ans)

st.markdown("---")
st.caption("Prototype by LegalLens AI — Use responsibly. Not legal advice.")