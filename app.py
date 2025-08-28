import os
import re
import io
import streamlit as st

# ML similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Optional parsers
try:
    import docx2txt
except Exception:
    docx2txt = None

try:
    import PyPDF2
except Exception:
    PyPDF2 = None


# ========= OpenAI: robust import + lazy client =========
def _sdk_info():
    try:
        import openai  # just to read version
        ver = getattr(openai, "__version__", "unknown")
        return True, ver
    except Exception:
        return False, None


_OPENAI_INSTALLED, _OPENAI_VERSION = _sdk_info()


def get_openai_client():
    """
    Build a client *when needed*.
    Returns (client, err_msg). If err_msg is not None, show it to the user.
    """
    if not _OPENAI_INSTALLED:
        return None, "OpenAI SDK not installed. Run: pip install openai"

    try:
        from openai import OpenAI
    except Exception as e:
        return None, f"OpenAI import failed: {e}"

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None, "OPENAI_API_KEY not set in this session. In PowerShell: $env:OPENAI_API_KEY=\"sk-...\""

    try:
        client = OpenAI()  # reads env var
        return client, None
    except Exception as e:
        return None, f"Failed to initialize OpenAI client: {e}"


# ========= Helpers =========
def extract_text_from_upload(uploaded_file):
    """Extract plain text from uploaded .txt/.docx/.pdf; fallback to best-effort decode."""
    if uploaded_file is None:
        return ""
    name = uploaded_file.name.lower()
    content = uploaded_file.read()
    try:
        uploaded_file.seek(0)
    except Exception:
        pass

    # .txt
    if name.endswith(".txt"):
        for enc in ("utf-8", "latin-1"):
            try:
                return content.decode(enc, errors="ignore")
            except Exception:
                continue
        return ""

    # .docx
    if name.endswith(".docx") and docx2txt is not None:
        try:
            tmp = "temp_upload.docx"
            with open(tmp, "wb") as f:
                f.write(content)
            text = docx2txt.process(tmp) or ""
            try:
                os.remove(tmp)
            except Exception:
                pass
            return text
        except Exception:
            pass  # fall through to generic decode

    # .pdf
    if name.endswith(".pdf") and PyPDF2 is not None:
        try:
            reader = PyPDF2.PdfReader(io.BytesIO(content))
            pages = []
            for p in reader.pages:
                try:
                    pages.append(p.extract_text() or "")
                except Exception:
                    pages.append("")
            return "\n".join(pages)
        except Exception:
            pass  # fall through to generic decode

    # Fallback
    for enc in ("utf-8", "latin-1"):
        try:
            return content.decode(enc, errors="ignore")
        except Exception:
            continue
    return ""


def clean_text(t: str) -> str:
    t = t or ""
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def keyword_coverage(resume_text: str, jd_text: str):
    resume = resume_text.lower()
    words = [w for w in re.findall(r"[a-zA-Z][a-zA-Z\+\#\.\-]{1,}", jd_text.lower()) if len(w) > 2]
    unique = sorted(set(words))
    found = [w for w in unique if w in resume]
    coverage = (len(found) / max(1, len(unique))) * 100.0
    return coverage, found, unique


def similarity_score(resume_text: str, jd_text: str) -> float:
    docs = [resume_text, jd_text]
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    try:
        X = vec.fit_transform(docs)
        sim = cosine_similarity(X[0:1], X[1:2])[0, 0] * 100.0
        return float(sim)
    except Exception:
        return 0.0


def detect_ats_issues(resume_text: str):
    issues = []
    if len(resume_text) < 500:
        issues.append("Resume seems short. Add detail and measurable achievements.")
    if re.search(r"\t", resume_text):
        issues.append("Avoid TAB characters; some ATS parsers misread complex formatting.")
    if not re.search(r"@", resume_text):
        issues.append("Missing contact email.")
    if not re.search(r"\d{10}", resume_text) and not re.search(r"\(\d{3}\)\s*\d{3}-\d{4}", resume_text):
        issues.append("Missing phone number.")
    if not re.search(r"education|bachelor|master|university", resume_text, flags=re.I):
        issues.append("Education section not detected (use heading 'Education').")
    if not re.search(r"experience|work history|employment", resume_text, flags=re.I):
        issues.append("Experience section not detected (use heading 'Experience').")
    return issues


def generate_suggestions(resume_text: str, jd_text: str, found_keywords):
    suggestions = []
    _, _, all_keywords = keyword_coverage(resume_text, jd_text)
    missing = [w for w in all_keywords if w not in found_keywords][:15]
    if missing:
        suggestions.append(f"Add relevant JD keywords (missing examples: {', '.join(missing[:10])}).")
    if not re.search(r"\b(\d+%|\d{2,})\b", resume_text):
        suggestions.append("Quantify achievements (e.g., 'Improved accuracy by 12%', 'Processed 1M+ rows').")
    if resume_text.count("•") < 3 and resume_text.count("- ") < 3:
        suggestions.append("Use concise bullet points with action verbs (Built, Led, Automated, Reduced).")
    suggestions.append("Tailor the top 5 bullets to the role’s must-have skills and responsibilities.")
    return suggestions


def example_bullets():
    return [
        "Built and deployed an end-to-end ML pipeline in Python (scikit-learn), improving prediction accuracy by 12%.",
        "Implemented feature engineering and GridSearchCV tuning, reducing inference latency by 30%.",
        "Developed KPI dashboards (Streamlit) and communicated insights to cross-functional stakeholders.",
        "Automated data ingestion from APIs; improved data quality and saved ~6 hrs/week of manual work."
    ]


# ========= UI =========
st.set_page_config(page_title="AI Resume & Job Matcher", page_icon="💼")
st.title("💼 AI Resume & Job Matcher — Starter (with diagnostics)")
st.write("Upload your resume, paste a JD, get a **match score**, **keyword coverage**, **ATS checks**, and **tips**. Optional: **AI rewrite** with OpenAI.")

with st.sidebar:
    st.header("How to use")
    st.markdown(
        "1) Upload a resume (.pdf/.docx/.txt) **or** paste text\n"
        "2) Paste the Job Description (JD)\n"
        "3) Click **Score My Resume**\n"
        "4) (Optional) Use **AI Rewrite** to tailor your resume"
    )
    st.subheader("AI Status")
    st.write(f"OpenAI SDK installed: **{_OPENAI_INSTALLED}**" + (f" (v{_OPENAI_VERSION})" if _OPENAI_VERSION else ""))
    st.write("API key detected: **" + ("Yes" if os.getenv("OPENAI_API_KEY") else "No") + "**")
    st.caption('If "No", set it in this terminal:\n$env:OPENAI_API_KEY="sk-..." and restart the app.')

# Inputs
resume_file = st.file_uploader("Upload Resume (.pdf, .docx, .txt)", type=["pdf", "docx", "txt"])
resume_text_manual = st.text_area("Or paste resume text here", height=180, placeholder="Paste your resume text…")
jd_text = st.text_area("Paste Job Description (JD) here", height=220, placeholder="Paste the target job description…")

# Keep processed texts in session so Rewrite works after scoring
if "rt" not in st.session_state:
    st.session_state.rt = ""
if "jd" not in st.session_state:
    st.session_state.jd = ""

if st.button("🔎 Score My Resume", type="primary"):
    with st.spinner("Analyzing…"):
        rt = extract_text_from_upload(resume_file) if resume_file else ""
        if not rt and resume_text_manual:
            rt = resume_text_manual

        rt = clean_text(rt)
        jd = clean_text(jd_text)

        if not rt or not jd:
            st.error("Please provide both a resume and a job description (JD).")
        else:
            # store so rewrite can use them
            st.session_state.rt = rt
            st.session_state.jd = jd

            sim = similarity_score(rt, jd)
            coverage, found, _ = keyword_coverage(rt, jd)
            issues = detect_ats_issues(rt)
            sugg = generate_suggestions(rt, jd, found)

            st.subheader("Results")
            c1, c2 = st.columns(2)
            c1.metric("Resume ↔ JD Match (TF-IDF)", f"{sim:.1f}%")
            c2.metric("Keyword Coverage", f"{coverage:.1f}%")

            st.markdown("**Keywords detected in your resume (from JD)**")
            st.write(", ".join(found[:100]) if found else "_No JD keywords detected in resume text._")

            if issues:
                st.markdown("### ATS-style Checks (basic)")
                for i in issues:
                    st.warning(i)

            st.markdown("### Suggestions to Improve")
            for s in sugg:
                st.info("• " + s)

            st.markdown("### Example Bullet Points You Can Adapt")
            for b in example_bullets():
                st.write(f"- {b}")

            with st.expander("Show extracted resume text (debug)"):
                st.text(rt[:6000])

# ========= AI Rewrite Section (uses session-state texts) =========
st.markdown("---")
st.markdown("### ✨ AI Rewrite (optional)")

if not st.session_state.rt or not st.session_state.jd:
    st.info("Add a resume and JD above, then click **Score My Resume** first. The rewrite uses those texts.")
else:
    if st.button("Rewrite my resume for this JD"):
        client, err = get_openai_client()
        if err:
            st.error(err)
        else:
            with st.spinner("Rewriting…"):
                prompt = f"""
You are an expert resume editor. Rewrite the RESUME to better match the JOB DESCRIPTION.
- Keep only truthful content (no fake experience).
- Keep ATS-friendly formatting (plain text, clear section headings).
- Add missing keywords naturally.
- Quantify achievements where possible.
- Output plain text only (no markdown).

--- RESUME ---
{st.session_state.rt}

--- JOB DESCRIPTION ---
{st.session_state.jd}
"""
                try:
                    resp = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.3,
                    )
                    revised = resp.choices[0].message.content.strip()
                    st.download_button("Download Revised Resume (TXT)", revised, file_name="revised_resume.txt")
                    st.text_area("Revised Resume (preview)", revised, height=350)
                except Exception as e:
                    st.error(f"Rewrite failed: {e}")

st.markdown("---")
st.caption("Starter template. Next: persistence (SQLite/Supabase), multi-JD ranking, and Stripe paywall.")
