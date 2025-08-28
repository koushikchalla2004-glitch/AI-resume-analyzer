import os
import streamlit as st
from dotenv import load_dotenv

st.set_page_config(page_title="US University Recommender", page_icon="🎓", layout="wide")

# Load .env for local
load_dotenv()

# Always try Streamlit Cloud secrets; harmless locally (will just skip if not present)
try:
    key = st.secrets.get("SCORECARD_API_KEY", None)
    if key:
        os.environ["SCORECARD_API_KEY"] = key
except Exception:
    pass

configure_api_key()

# ---- rest of your imports (now env is ready) ----
from src.scorecard_api import ScorecardClient
from src.cip_map import CIP_MAP
from src.lor_sop_nlp import score_document
from src.acceptance import acceptance_probability
from src.utils import USD, safe

st.title("🎓 US University Recommender – Real Data + NLP")

with st.sidebar:
    st.header("Your Profile")
    name = st.text_input("Name", placeholder="Your name")
    course = st.text_input("Preferred course / major", value="Data Science")
    loc_mode = st.radio("Location filter", ["State", "City", "ZIP radius", "Anywhere"], index=0)
    state = city = zipc = radius = None
    if loc_mode == "State":
        state = st.text_input("State (2-letter)", value="TX", max_chars=2)
    elif loc_mode == "City":
        city = st.text_input("City", value="Dallas")
        state = st.text_input("State (2-letter)", value="TX", max_chars=2)
    elif loc_mode == "ZIP radius":
        zipc = st.text_input("ZIP code", value="75201")
        radius = st.selectbox("Distance", ["10mi", "25mi", "50mi", "100mi"], index=1)

    st.markdown("---")
    st.subheader("Budget (2 years)")
    budget = st.number_input("Total budget for 2 years (USD)", min_value=10000, step=1000, value=80000)

    st.markdown("---")
    st.subheader("Academics")
    cgpa = st.number_input("CGPA (out of 4.0)", min_value=0.0, max_value=4.0, step=0.01, value=3.6)
    gre = st.number_input("GRE (optional)", min_value=0, max_value=340, step=1, value=320)
    ielts = st.number_input("IELTS (optional)", min_value=0.0, max_value=9.0, step=0.5, value=7.0)

    st.markdown("---")
    st.subheader("LOR / SOP Upload (PDF or .txt)")
    lor_sop_kind = st.radio("Document type", ["SOP", "LOR"], index=0)
    upload = st.file_uploader("Upload LOR/SOP (PDF or .txt)", type=["pdf", "txt"])

    st.markdown("---")
    st.caption("Costs use College Scorecard tuition + on-campus room/board + other on-campus expenses; books shown separately.")

if "go" not in st.session_state:
    st.session_state["go"] = False

colL, colR = st.columns([1, 2])
with colL:
    if st.button("🔎 Find Universities", use_container_width=True):
        st.session_state["go"] = True
with colR:
    st.info("Tip: you can export results at the bottom as CSV.")

if st.session_state.get("go"):
    client = ScorecardClient()

    key = (course or "").strip().lower()
    cip4 = CIP_MAP.get(key)
    title_contains = None if cip4 else course
    zip_radius = (zipc, radius) if (zipc and radius) else None

    results = []
    for page in range(0, 3):
        data = client.search(
            state=state,
            city=city,
            zip_radius=zip_radius,
            cip4=cip4,
            title_contains=title_contains,
            page=page,
        )
        page_results = data.get("results", [])
        results.extend(page_results)
        if len(page_results) < 100:
            break

    if not results:
        st.warning("No matching institutions found. Try broadening location or course.")
        st.stop()

    lor_score = 0.5
    lor_details = {}
    if upload is not None:
        bytes_data = upload.read()
        lor_score, lor_details = score_document(bytes_data, upload.name, kind_hint=lor_sop_kind.lower())

    rows = []
    for r in results:
        name_ = safe(r.get("school.name"), "")
        city_ = safe(r.get("school.city"), "")
        state_ = safe(r.get("school.state"), "")
        url_ = safe(r.get("school.school_url"), "")

        base_rate = r.get("latest.admissions.admission_rate.overall")
        in_tuition = r.get("latest.cost.tuition.in_state") or 0
        out_tuition = r.get("latest.cost.tuition.out_of_state") or 0
        room = r.get("latest.cost.roomboard.oncampus") or 0
        other = r.get("latest.cost.other_on_campus") or 0
        books = r.get("latest.cost.booksupply") or 0

        tuition = out_tuition if out_tuition else in_tuition
        living = (room or 0) + (other or 0)

        per_year = (tuition or 0) + living
        two_year = 2 * per_year
        meets_budget = two_year <= budget

        acc = acceptance_probability(
            base_rate,
            cgpa=cgpa,
            gre=(gre or None),
            ielts=(ielts or None),
            lor_sop=lor_score,
        )

        rows.append(
            {
                "Institution": name_,
                "City": city_,
                "State": state_,
                "URL": url_,
                "Tuition (yr)": tuition,
                "Living (yr)": living,
                "Books (yr)": books,
                "Total (yr)": per_year,
                "Total (2y)": two_year,
                "Baseline admit": base_rate,
                "Your admit %": round(100 * acc, 1),
                "Within budget?": "✅" if meets_budget else "—",
            }
        )

    df = pd.DataFrame(rows)
    df.sort_values(["Within budget?", "Total (2y)", "Your admit %"], ascending=[False, True, False], inplace=True)

    st.subheader("Results")
    st.dataframe(
        df.assign(
            **{
                "Tuition (yr)": df["Tuition (yr)"].map(lambda x: USD.format(x) if x else "—"),
                "Living (yr)": df["Living (yr)"].map(lambda x: USD.format)
            }
        )
    ) 
