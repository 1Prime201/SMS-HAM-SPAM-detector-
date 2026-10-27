"""
Streamlit Dashboard for XAI Spam Classifier.
High-fidelity UI with confidence gauge, SHAP text plots, and threshold tuning.
"""

import streamlit as st
import requests
import plotly.graph_objects as go

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
API_URL = "http://127.0.0.1:8000/"

st.set_page_config(
    page_title="XAI Spam Classifier",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.main-header {
    text-align: center;
    padding: 1.5rem 0 0.5rem 0;
}
.main-header h1 {
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}
.main-header p {
    color: #8892b0;
    font-size: 1.05rem;
    font-weight: 400;
}

.result-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border-radius: 16px;
    padding: 1.8rem;
    margin: 1rem 0;
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}

.spam-badge {
    display: inline-block;
    padding: 0.4rem 1.2rem;
    border-radius: 24px;
    font-weight: 700;
    font-size: 1.1rem;
    letter-spacing: 0.5px;
}
.spam-badge.spam {
    background: linear-gradient(135deg, #ff416c, #ff4b2b);
    color: white;
}
.spam-badge.ham {
    background: linear-gradient(135deg, #11998e, #38ef7d);
    color: #0a2e1a;
}

.shap-container {
    background: #0d1117;
    border-radius: 12px;
    padding: 1.2rem;
    margin: 1rem 0;
    border: 1px solid rgba(255,255,255,0.06);
    overflow-x: auto;
}
.shap-token {
    display: inline-block;
    padding: 3px 6px;
    margin: 2px 1px;
    border-radius: 4px;
    font-family: 'Courier New', monospace;
    font-size: 0.95rem;
    font-weight: 500;
    transition: transform 0.15s ease;
}
.shap-token:hover {
    transform: scale(1.15);
    cursor: pointer;
}

.sidebar-section h3 {
    font-size: 1rem;
    font-weight: 600;
    color: #ccd6f6;
    margin-bottom: 0.3rem;
}

.metrics-row {
    display: flex;
    gap: 1rem;
    margin: 1rem 0;
}
.metric-box {
    flex: 1;
    background: rgba(255,255,255,0.04);
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.06);
}
.metric-box .value {
    font-size: 1.8rem;
    font-weight: 700;
}
.metric-box .label {
    font-size: 0.8rem;
    color: #8892b0;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.examples-btn {
    width: 100%;
    padding: 0.5rem;
    margin: 0.25rem 0;
    border-radius: 8px;
    border: 1px solid rgba(255,255,255,0.1);
    background: rgba(255,255,255,0.03);
    color: #ccd6f6;
    cursor: pointer;
    transition: all 0.2s ease;
    text-align: left;
    font-size: 0.85rem;
}
.examples-btn:hover {
    background: rgba(102, 126, 234, 0.15);
    border-color: #667eea;
}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Cached helpers
# ──────────────────────────────────────────────

def check_backend_health():
    """Check if FastAPI backend is reachable."""
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def call_explain(text: str) -> dict | None:
    """Call the /explain endpoint."""
    try:
        r = requests.post(f"{API_URL}/explain", json={"text": text}, timeout=60)
        if r.status_code == 200:
            return r.json()
        else:
            st.error(f"API Error {r.status_code}: {r.text}")
            return None
    except requests.ConnectionError:
        st.error("Cannot connect to backend. Make sure `python backend.py` is running.")
        return None
    except Exception as e:
        st.error(f"Request failed: {e}")
        return None


# ──────────────────────────────────────────────
# Confidence Gauge (Plotly)
# ──────────────────────────────────────────────

def render_confidence_gauge(spam_prob: float, threshold: float):
    """Render a dynamic confidence gauge using Plotly."""
    is_spam = spam_prob >= threshold
    color = "#ff416c" if is_spam else "#38ef7d"
    label_text = "SPAM" if is_spam else "HAM"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=spam_prob * 100,
        number={"suffix": "%", "font": {"size": 48, "color": color}},
        title={"text": f"Spam Confidence — <b>{label_text}</b>",
               "font": {"size": 18, "color": "#ccd6f6"}},
        delta={"reference": threshold * 100, "increasing": {"color": "#ff416c"},
               "decreasing": {"color": "#38ef7d"}, "suffix": "%",
               "font": {"size": 14}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1,
                     "tickcolor": "#333", "dtick": 10,
                     "tickfont": {"color": "#8892b0"}},
            "bar": {"color": color, "thickness": 0.3},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, threshold * 100], "color": "rgba(56, 239, 125, 0.08)"},
                {"range": [threshold * 100, 100], "color": "rgba(255, 65, 108, 0.08)"},
            ],
            "threshold": {
                "line": {"color": "#667eea", "width": 3},
                "thickness": 0.8,
                "value": threshold * 100,
            },
        },
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#ccd6f6"},
        height=300,
        margin=dict(l=30, r=30, t=60, b=20),
    )
    return fig


# ──────────────────────────────────────────────
# SHAP Text Visualization
# ──────────────────────────────────────────────

def render_shap_html(tokens: list[dict], base_value: float):
    """Render interactive SHAP text plot as styled HTML."""
    if not tokens:
        return "<p style='color:#8892b0;'>No token data available.</p>"

    max_abs = max(abs(t["shap_value"]) for t in tokens) or 1.0
    html_parts = []

    for t in tokens:
        val = t["shap_value"]
        norm = val / max_abs  # -1 to 1
        if val > 0:
            # Red tones for spam-contributing tokens
            r, g, b = 255, 65, 108
            opacity = min(abs(norm) * 0.7 + 0.1, 0.85)
        elif val < 0:
            # Green tones for ham-contributing tokens
            r, g, b = 56, 239, 125
            opacity = min(abs(norm) * 0.7 + 0.1, 0.85)
        else:
            r, g, b = 136, 146, 176
            opacity = 0.1

        bg = f"rgba({r},{g},{b},{opacity:.2f})"
        border_clr = f"rgba({r},{g},{b},{min(opacity+0.2, 1):.2f})"
        title = f"{t['token']}: SHAP = {val:+.4f}"

        html_parts.append(
            f'<span class="shap-token" style="background:{bg}; '
            f'border: 1px solid {border_clr};" '
            f'title="{title}">{t["token"]}</span>'
        )

    base_info = (
        f'<div style="margin-bottom:0.6rem; color:#8892b0; font-size:0.82rem;">'
        f'Base value (model bias toward spam): <b>{base_value:.4f}</b></div>'
    )
    legend = (
        '<div style="margin-top:0.8rem; font-size:0.78rem; color:#8892b0;">'
        '<span style="color:#ff416c;">■</span> Pushes toward Spam &nbsp;&nbsp;'
        '<span style="color:#38ef7d;">■</span> Pushes toward Ham &nbsp;&nbsp;'
        '| Hover tokens for exact SHAP values</div>'
    )
    return base_info + "".join(html_parts) + legend


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Controls")
    st.markdown("---")

    # Threshold Tuning
    st.markdown("### 🎯 Spam Threshold")
    if "threshold" not in st.session_state:
        st.session_state.threshold = 0.50

    st.session_state.threshold = st.slider(
        "Sensitivity",
        min_value=0.05,
        max_value=0.95,
        value=st.session_state.threshold,
        step=0.05,
        help="Lower = more aggressive spam detection. Higher = fewer false positives.",
    )

    th = st.session_state.threshold
    if th < 0.3:
        st.caption("🔴 Very aggressive — may flag legitimate messages")
    elif th < 0.5:
        st.caption("🟠 Moderately aggressive")
    elif th == 0.5:
        st.caption("🟢 Balanced (default)")
    elif th < 0.75:
        st.caption("🔵 Conservative — some spam may slip through")
    else:
        st.caption("⚪ Very conservative — only flags obvious spam")

    st.markdown("---")

    # Example messages
    st.markdown("### 📝 Quick Examples")
    EXAMPLES = [
        "Congratulations! You've won a $1000 gift card. Click here to claim now!",
        "Hey, are we still meeting for lunch tomorrow at noon?",
        "URGENT: Your bank account has been compromised. Verify your identity NOW.",
        "Mom says dinner is at 7. Don't be late!",
        "FREE entry in a weekly competition. Text WIN to 80085 to claim your prize!",
        "Can you pick up some milk on your way home?",
    ]

    for i, ex in enumerate(EXAMPLES):
        if st.button(ex[:55] + "..." if len(ex) > 55 else ex, key=f"ex_{i}", use_container_width=True):
            st.session_state.input_text = ex

    st.markdown("---")

    # Backend Status
    st.markdown("### 🔌 Backend Status")
    if check_backend_health():
        st.success("Connected to API", icon="✅")
    else:
        st.error("Backend offline", icon="❌")
        st.caption("Run: `python backend.py`")

    st.markdown("---")
    st.markdown(
        '<div style="text-align:center;color:#555;font-size:0.75rem;">'
        'Built with DistilBERT · SHAP · FastAPI · Streamlit</div>',
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────
# Main Content
# ──────────────────────────────────────────────

st.markdown(
    '<div class="main-header">'
    '<h1>🛡️ XAI Spam Classifier</h1>'
    '<p>DistilBERT-powered spam detection with word-level SHAP explanations</p>'
    '</div>',
    unsafe_allow_html=True,
)

# Input area
col_input, col_btn = st.columns([5, 1])

with col_input:
    if "input_text" not in st.session_state:
        st.session_state.input_text = ""

    user_text = st.text_area(
        "Enter SMS message",
        value=st.session_state.input_text,
        height=100,
        placeholder="Type or paste an SMS message to analyze...",
        label_visibility="collapsed",
    )

with col_btn:
    st.markdown("<br>", unsafe_allow_html=True)
    analyze_clicked = st.button("🔍 Analyze", type="primary", use_container_width=True)

# ──────────────────────────────────────────────
# Results
# ──────────────────────────────────────────────

if analyze_clicked and user_text.strip():
    with st.spinner("🧠 Running model inference & SHAP analysis..."):
        result = call_explain(user_text.strip())

    if result:
        spam_prob = result["spam_probability"]
        ham_prob = result["ham_probability"]
        threshold = st.session_state.threshold
        is_spam = spam_prob >= threshold
        final_label = "Spam" if is_spam else "Ham (Not Spam)"

        # ── Result Card ──
        st.markdown('<div class="result-card">', unsafe_allow_html=True)

        badge_class = "spam" if is_spam else "ham"
        st.markdown(
            f'<span class="spam-badge {badge_class}">{final_label}</span>',
            unsafe_allow_html=True,
        )

        # Metrics row
        st.markdown(
            f'<div class="metrics-row">'
            f'<div class="metric-box">'
            f'<div class="value" style="color:#ff416c;">{spam_prob*100:.1f}%</div>'
            f'<div class="label">Spam Probability</div></div>'
            f'<div class="metric-box">'
            f'<div class="value" style="color:#38ef7d;">{ham_prob*100:.1f}%</div>'
            f'<div class="label">Ham Probability</div></div>'
            f'<div class="metric-box">'
            f'<div class="value" style="color:#667eea;">{threshold*100:.0f}%</div>'
            f'<div class="label">Threshold</div></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.markdown('</div>', unsafe_allow_html=True)

        # ── Gauge + SHAP side by side ──
        col_gauge, col_shap = st.columns([1, 1])

        with col_gauge:
            st.markdown("#### 📊 Confidence Gauge")
            fig = render_confidence_gauge(spam_prob, threshold)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        with col_shap:
            st.markdown("#### 🔬 Word-Level SHAP Explanations")
            shap_html = render_shap_html(result["tokens"], result["base_value"])
            st.markdown(
                f'<div class="shap-container">{shap_html}</div>',
                unsafe_allow_html=True,
            )

        # ── Detailed Token Table ──
        with st.expander("📋 Detailed Token Attribution Table", expanded=False):
            import pandas as pd
            df = pd.DataFrame(result["tokens"])
            df.columns = ["Token", "SHAP Value"]
            df["Direction"] = df["SHAP Value"].apply(
                lambda x: "→ Spam" if x > 0 else ("→ Ham" if x < 0 else "Neutral")
            )
            df["Abs Impact"] = df["SHAP Value"].abs()
            df = df.sort_values("Abs Impact", ascending=False)
            st.dataframe(df, use_container_width=True, hide_index=True)

elif analyze_clicked:
    st.warning("Please enter a message to analyze.", icon="⚠️")
