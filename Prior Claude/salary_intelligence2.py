import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import Ridge
from statsmodels.regression.quantile_regression import QuantReg
import statsmodels.api as sm
from datetime import datetime
from scipy.stats import zscore
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from fpdf import FPDF
import io

# === Styled PDF Class (Amazon Color Scheme) ===
class StyledPDF(FPDF):
    def header(self):
        self.set_fill_color(0, 51, 102)  # Navy Blue
        self.set_text_color(255, 255, 255)
        self.set_font("Arial", "B", 16)
        self.cell(0, 12, "Salary Intelligence Report", ln=True, align="C", fill=True)
        self.ln(4)

    def section_title(self, title):
        self.set_fill_color(255, 153, 0)  # Amazon Orange
        self.set_text_color(0, 0, 0)
        self.set_font("Arial", "B", 12)
        self.cell(0, 10, title, ln=True, fill=True)
        self.ln(2)

    def section_body(self, text):
        self.set_font("Arial", "", 11)
        self.set_text_color(0, 0, 0)
        self.multi_cell(0, 8, text)
        self.ln()

# === Configuration ===
DFW_COL_INDEX = 98
DEFAULT_COL_INDICES = {
    'California': 142,
    'Colorado': 105,
    'New York': 139,
    'Illinois': 108,
    'Washington': 130,
    'Texas': 98
}
PRIORITY_COMPETITORS = ["Milwaukee Tool", "DeWalt", "Snap-on"]
CONFIDENCE_THRESHOLDS = {'high': 20, 'medium': 10}

# === Streamlit Setup ===
st.set_page_config(page_title="Salary Intelligence Platform", layout="wide")
st.title("💼 Salary Intelligence Platform")

# === Sidebar ===
st.sidebar.title("Model Settings")
data_source = st.sidebar.radio("Select Data Source", ["Simulated Data", "Upload CSV"])
role = st.sidebar.selectbox("Target Role", ["Mechanical Engineer - P1", "Mechanical Engineer - P2", "Product Manager"])

def normalize_title(title):
    title = title.lower()
    if "principal" in title or "p6" in title:
        return "P6"
    elif "staff" in title or "p5" in title:
        return "P5"
    elif "lead" in title or "p4" in title:
        return "P4"
    elif "senior" in title or "sr" in title or "p3" in title:
        return "P3"
    elif "mid" in title or "p2" in title:
        return "P2"
    else:
        return "P1"

def cluster_job_titles(df, num_clusters=5):
    titles = df['role'].fillna("").astype(str)
    tfidf = TfidfVectorizer(stop_words='english')
    X_tfidf = tfidf.fit_transform(titles)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    df['title_cluster'] = kmeans.fit_predict(X_tfidf)
    return df, kmeans

def get_confidence_level(n):
    if n >= CONFIDENCE_THRESHOLDS['high']:
        return "⭐⭐⭐⭐⭐ High"
    elif n >= CONFIDENCE_THRESHOLDS['medium']:
        return "⭐⭐⭐ Medium"
    else:
        return "⭐ Low"

def generate_pdf(role, states, companies, date, p25, p50, p75, sample_size, df_head, confidence, cluster_table=None):
    pdf = StyledPDF()
    pdf.add_page()
    pdf.section_title("Overview")
    pdf.section_body(f"Role: {role}")
    pdf.section_body(f"Markets: {', '.join(states)}")
    pdf.section_body(f"Competitors: {', '.join(companies)}")
    pdf.section_body(f"Date of Analysis: {date}")
    pdf.section_body(f"Model Confidence: {confidence}")

    pdf.section_title("Model Results")
    pdf.section_body(f"Sample Size: {sample_size}")
    pdf.section_body(f"P25 Estimate: ${p25:,.0f}")
    pdf.section_body(f"Median Estimate: ${p50:,.0f}")
    pdf.section_body(f"P75 Estimate: ${p75:,.0f}")

    pdf.section_title("Sample Job Data")
    for _, row in df_head.iterrows():
        txt = f"{row['state']} - {row['company']} - {row['role']} - ${row['avg_salary']:,.0f}"
        pdf.section_body(txt)

    if cluster_table is not None:
        pdf.section_title("Title Cluster Summary")
        for _, row in cluster_table.iterrows():
            pdf.section_body(f"Cluster {row['title_cluster']}: {row['sample_titles']}")

    output = io.BytesIO()
    pdf.output(output)
    return output

# === Load Data ===
if data_source == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        df['avg_salary'] = (df['min_salary'] + df['max_salary']) / 2
        df['col_index'] = df['state'].map(DEFAULT_COL_INDICES).fillna(100)
        df['sample_size'] = 1
        df['normalized_role'] = df['role'].apply(normalize_title)
    else:
        st.stop()
else:
    states = st.sidebar.multiselect("States", list(DEFAULT_COL_INDICES.keys()), default=["California", "Illinois"])
    np.random.seed(42)
    data = []
    for state in states:
        col_index = DEFAULT_COL_INDICES[state]
        for _ in range(np.random.randint(5, 10)):
            base = 90000 + 5000 * ("P2" in role)
            salary = base * (col_index / 100) * np.random.normal(1.0, 0.05)
            data.append({
                'state': state,
                'company': np.random.choice(PRIORITY_COMPETITORS),
                'role': role,
                'min_salary': salary - 5000,
                'max_salary': salary + 5000,
                'avg_salary': salary,
                'col_index': col_index,
                'sample_size': 1,
                'normalized_role': normalize_title(role)
            })
    df = pd.DataFrame(data)

# === Modeling ===
st.sidebar.title("Location Adjustment")
override_col = st.sidebar.slider("Override DFW Cost of Living Index", 85, 150, DFW_COL_INDEX)

df['z'] = zscore(df['avg_salary'])
df_filtered = df[(df['z'] > -2) & (df['z'] < 2)].copy()

use_clustering = st.sidebar.checkbox("Enable Job Title Clustering", value=True)
if use_clustering:
    df_filtered, _ = cluster_job_titles(df_filtered)

features = ['col_index', 'sample_size']
if use_clustering and 'title_cluster' in df_filtered.columns:
    df_filtered = pd.get_dummies(df_filtered, columns=['title_cluster'], drop_first=True)
    features += [col for col in df_filtered.columns if col.startswith('title_cluster_')]

X = sm.add_constant(df_filtered[features])
y = df_filtered['avg_salary']
results_25 = QuantReg(y, X).fit(q=0.25)
results_50 = QuantReg(y, X).fit(q=0.5)
results_75 = QuantReg(y, X).fit(q=0.75)

dfw_input = pd.DataFrame([[override_col, df_filtered['sample_size'].mean()] + [0]*(len(features)-2)], columns=features)
dfw_X = sm.add_constant(dfw_input)

pred_25 = results_25.predict(dfw_X)[0]
pred_50 = results_50.predict(dfw_X)[0]
pred_75 = results_75.predict(dfw_X)[0]
confidence_level = get_confidence_level(len(df_filtered))

# === Display Output ===
st.subheader("📈 Salary Estimate Summary")
st.success(f"Model Confidence: {confidence_level}")
st.metric("P25 Estimate", f"${pred_25:,.0f}")
st.metric("Median Estimate", f"${pred_50:,.0f}")
st.metric("P75 Estimate", f"${pred_75:,.0f}")

st.markdown("---")
st.subheader("📊 Market Salary Distribution")
fig = px.box(df_filtered, x='state', y='avg_salary', points="all", color='state')
st.plotly_chart(fig, use_container_width=True)

if use_clustering and 'title_cluster' in df_filtered.columns:
    st.subheader("🔎 Title Clustering")
    view = st.radio("View:", ["Bar Chart", "Sample Titles Table"])
    if view == "Bar Chart":
        cluster_counts = df_filtered['title_cluster'].value_counts().reset_index()
        cluster_counts.columns = ['title_cluster', 'count']
        st.plotly_chart(px.bar(cluster_counts, x='title_cluster', y='count'), use_container_width=True)
    else:
        sample_titles = (
            df_filtered.groupby('title_cluster')['role']
            .apply(lambda x: ', '.join(x.unique()[:3]))
            .reset_index().rename(columns={'role': 'sample_titles'})
        )
        st.dataframe(sample_titles)

st.markdown("---")
st.subheader("📄 Export Market Report")
if st.button("Generate PDF"):
    sample_preview = df_filtered[['state', 'company', 'role', 'avg_salary']].head(10)
    cluster_info_df = None
    if use_clustering and 'title_cluster' in df_filtered.columns:
        cluster_info_df = (
            df_filtered.groupby('title_cluster')['role']
            .apply(lambda x: ', '.join(x.unique()[:3]))
            .reset_index().rename(columns={'role': 'sample_titles'})
        )
    date_now = datetime.now().strftime("%Y-%m-%d")
    pdf_bytes = generate_pdf(role, df_filtered['state'].unique(), df_filtered['company'].unique(),
                             date_now, pred_25, pred_50, pred_75,
                             len(df_filtered), sample_preview, confidence_level, cluster_info_df)
    st.download_button("⬇️ Download Report", data=pdf_bytes, file_name=f"Market_Intelligence_{role.replace(' ', '_')}.pdf")
