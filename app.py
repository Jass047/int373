# Workplace Injury Prediction - Streamlit App
# INT234 Predictive Analytics Project - Jaskaran Singh

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import chi2_contingency
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(page_title="Workplace Injury Predictor", page_icon="🦺", layout="wide")

# ------------------------------------------------------------------
# Colours used in all charts
# ------------------------------------------------------------------
ORANGE = "#E8590C"
NAVY = "#1B3A5B"
GREEN = "#2B8A3E"
AMBER = "#F59F00"
RED = "#C92A2A"
GREY = "#868E96"

st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1200px;}
.hero {background: linear-gradient(120deg, #1B3A5B 0%, #24527A 100%); color: white;
       padding: 2rem 2.2rem; border-radius: 14px; margin-bottom: 1.2rem;}
.hero h1 {color: white; margin: 0 0 .4rem 0; font-size: 2.1rem;}
.hero p {color: #DCE6F0; font-size: 1.05rem; margin: 0;}
.card {background: #FFFFFF; border: 1px solid #E3E8EE; border-radius: 12px; padding: 1.1rem 1.2rem;
       height: 100%; box-shadow: 0 1px 3px rgba(16,24,40,.06);}
.card .label {color: #5F6B7A; font-size: .85rem; text-transform: uppercase; letter-spacing: .04em;}
.card .value {color: #1B3A5B; font-size: 1.9rem; font-weight: 700; margin-top: .2rem;}
.card .value.text {font-size: 1.3rem; line-height: 1.3; margin-top: .45rem; word-break: normal; overflow-wrap: normal;}
.card .note {color: #5F6B7A; font-size: .85rem; margin-top: .2rem;}
.step {background: #FFF4E6; border-left: 5px solid #E8590C; border-radius: 8px; padding: .8rem 1rem; margin-bottom: .6rem;}
.explain {background: #F1F5F9; border-radius: 8px; padding: .7rem 1rem; color: #334155; font-size: .95rem; margin: .2rem 0 1.2rem 0;}
.risk {border-radius: 14px; padding: 1.4rem 1.6rem; color: white; text-align: center;}
.risk .big {font-size: 2.8rem; font-weight: 800; line-height: 1.1;}
.risk .small {font-size: 1.05rem; opacity: .95;}
.tip {background: #EBFBEE; border-left: 5px solid #2B8A3E; border-radius: 8px; padding: .6rem 1rem; margin-bottom: .5rem;}
</style>
""", unsafe_allow_html=True)


def card(label, value, note=""):
    kind = "value text" if any(ch.isalpha() for ch in str(value)) else "value"
    st.markdown(f'<div class="card"><div class="label">{label}</div><div class="{kind}">{value}</div>'
                f'<div class="note">{note}</div></div>', unsafe_allow_html=True)


def explain(text):
    st.markdown(f'<div class="explain">💡 {text}</div>', unsafe_allow_html=True)


# ------------------------------------------------------------------
# 1. Load and clean the data (same steps as the notebook)
# ------------------------------------------------------------------
SECTOR = {'11': 'Agriculture', '21': 'Mining', '22': 'Utilities', '23': 'Construction',
          '31': 'Manufacturing', '32': 'Manufacturing', '33': 'Manufacturing',
          '42': 'Wholesale', '44': 'Retail', '45': 'Retail', '48': 'Transport', '49': 'Transport',
          '56': 'Admin & Waste', '62': 'Health Care', '72': 'Hotels & Food'}
EVENT = {'1': 'Violence', '2': 'Transport', '3': 'Fire/Explosion', '4': 'Fall',
         '5': 'Harmful Exposure', '6': 'Contact with Equipment', '7': 'Overexertion'}
SOURCE = {'1': 'Chemicals', '2': 'Containers', '3': 'Machinery', '4': 'Parts & Materials',
          '5': 'Persons/Animals', '6': 'Floors & Surfaces', '7': 'Tools & Equipment', '8': 'Vehicles'}
BODY = {'1': 'Head', '2': 'Neck', '3': 'Trunk', '4': 'Arm/Hand', '5': 'Leg/Foot', '8': 'Multiple'}
MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']


@st.cache_data(show_spinner="Loading the injury data...")
def load_data():
    data = pd.read_csv("data/injury_data.csv.gz", low_memory=False)
    data['Amputation'] = data['Amputation'].fillna(0)
    data = data.dropna(subset=['Primary NAICS'])
    data['EventDate'] = pd.to_datetime(data['EventDate'])
    data['Year'] = data['EventDate'].dt.year
    data['Month'] = data['EventDate'].dt.month
    data['Amputation_Flag'] = np.where(data['Amputation'] > 0, 1, 0)
    data['Sector'] = data['Primary NAICS'].astype(str).str[:2].map(SECTOR).fillna('Other')
    data['Event_Type'] = data['Event'].astype(str).str[0].map(EVENT).fillna('Other')
    data['Source_Type'] = data['Source'].astype(str).str[0].map(SOURCE).fillna('Other')
    data['Body_Part'] = data['Part of Body'].astype(str).str[0].map(BODY).fillna('Other')
    data['State'] = data['State'].str.title()
    return data


# ------------------------------------------------------------------
# 2. Train the models once and keep them in memory
# ------------------------------------------------------------------
FEATURES = ['Sector', 'Event_Type', 'Source_Type', 'Body_Part', 'Month']


@st.cache_resource(show_spinner="Training the 5 models (first time only, about 15 seconds)...")
def train_classifiers(data):
    sample = data.sample(20000, random_state=123)
    X = pd.get_dummies(sample[FEATURES], columns=FEATURES[:4])
    y = sample['Amputation_Flag']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=123)
    sc = StandardScaler()
    X_train_s = sc.fit_transform(X_train)
    X_test_s = sc.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=123),
        "SVM (RBF)": SVC(kernel="rbf"),
    }
    rows, matrices = [], {}
    for name, model in models.items():
        model.fit(X_train_s, y_train)
        pred = model.predict(X_test_s)
        rows.append([name, accuracy_score(y_test, pred), precision_score(y_test, pred),
                     recall_score(y_test, pred), f1_score(y_test, pred)])
        matrices[name] = confusion_matrix(y_test, pred)
    results = pd.DataFrame(rows, columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])
    return models, sc, list(X.columns), results, matrices


@st.cache_resource(show_spinner="Training the forecast model...")
def train_regression(data):
    monthly = data.groupby(['Sector', 'Year', 'Month']).size().reset_index(name='Injuries')
    monthly['Last_Month'] = monthly.groupby('Sector')['Injuries'].shift(1)
    monthly = monthly.dropna()
    le = LabelEncoder()
    monthly['Sector_Code'] = le.fit_transform(monthly['Sector'])
    X = monthly[['Last_Month', 'Month', 'Sector_Code']]
    y = monthly['Injuries']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
    scaler = StandardScaler()
    lr = LinearRegression().fit(scaler.fit_transform(X_train), y_train)
    y_pred = lr.predict(scaler.transform(X_test))
    metrics = {"MAE": mean_absolute_error(y_test, y_pred), "MSE": mean_squared_error(y_test, y_pred),
               "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)), "R2": r2_score(y_test, y_pred)}
    return lr, scaler, le, metrics, y_test.values, y_pred


@st.cache_resource(show_spinner="Grouping the states...")
def train_clusters(data):
    states = data.groupby('State').agg(Injuries=('Amputation_Flag', 'size'),
                                       Amputation_Rate=('Amputation_Flag', 'mean'),
                                       Manufacturing_Share=('Sector', lambda s: (s == 'Manufacturing').mean()))
    states = states[states['Injuries'] >= 100].copy()
    X = StandardScaler().fit_transform(states)
    wcss = [KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=123).fit(X).inertia_ for k in range(1, 9)]
    km = KMeans(n_clusters=3, init='k-means++', n_init=10, random_state=123)
    states['Cluster'] = km.fit_predict(X)
    # give each cluster a readable name based on its profile
    prof = states.groupby('Cluster').mean()
    big = prof['Injuries'].idxmax()
    rest = prof.drop(big)
    high = rest['Amputation_Rate'].idxmax()
    low = rest['Amputation_Rate'].idxmin()
    names = {big: "Very high volume", high: "Manufacturing-heavy", low: "Low manufacturing"}
    states['Group'] = states['Cluster'].map(names)
    return states, wcss


data = load_data()

# ------------------------------------------------------------------
# Sidebar navigation
# ------------------------------------------------------------------
st.sidebar.markdown("## 🦺 Injury Predictor")
st.sidebar.caption("INT234 Predictive Analytics Project")
page = st.sidebar.radio("Go to", ["🏠 Home", "📊 Explore the Data", "🎯 Amputation Risk Checker",
                                  "📈 Monthly Injury Forecast", "🏆 Model Results", "🗺️ State Groups",
                                  "ℹ️ About"], label_visibility="collapsed")
st.sidebar.divider()
st.sidebar.caption("Data: OSHA Severe Injury Reports, Jan 2015 – Nov 2025")
st.sidebar.caption("Made by Jaskaran Singh · LPU")


# ==================================================================
# HOME
# ==================================================================
if page == "🏠 Home":
    st.markdown("""<div class="hero"><h1>Predicting Severe Workplace Injuries</h1>
    <p>Using 1 lakh+ real injury reports to find where serious injuries happen
    and to predict which ones lead to an amputation.</p></div>""", unsafe_allow_html=True)

    results = train_classifiers(data)[3]
    best = results.sort_values('F1 Score', ascending=False).iloc[0]
    rate = data['Amputation_Flag'].mean() * 100
    manu_rate = data[data['Sector'] == 'Manufacturing']['Amputation_Flag'].mean() * 100

    c1, c2, c3, c4 = st.columns(4)
    with c1: card("Injury reports", f"{len(data):,}", "Jan 2015 – Nov 2025")
    with c2: card("Amputations", f"{rate:.0f}%", "about 1 in every 4 injuries")
    with c3: card("Riskiest sector", "Manufacturing", f"{manu_rate:.0f}% of its injuries are amputations")
    with c4: card("Best model", f"{best['Accuracy']*100:.1f}%", f"{best['Model']} accuracy")

    st.write("")
    left, right = st.columns([1.1, 1])
    with left:
        st.subheader("What can this app do?")
        st.markdown('<div class="step"><b>📊 Explore the Data</b> – see which industries, accidents and body parts are most affected.</div>', unsafe_allow_html=True)
        st.markdown('<div class="step"><b>🎯 Amputation Risk Checker</b> – choose a job situation and get its amputation risk with safety tips.</div>', unsafe_allow_html=True)
        st.markdown('<div class="step"><b>📈 Monthly Injury Forecast</b> – predict next month\'s injuries for any industry.</div>', unsafe_allow_html=True)
        st.markdown('<div class="step"><b>🏆 Model Results</b> and <b>🗺️ State Groups</b> – compare the machine learning models and see similar states.</div>', unsafe_allow_html=True)
        st.caption("👈 Use the menu on the left to move between pages.")
    with right:
        st.subheader("Key findings")
        st.markdown(f"""
- **Manufacturing** has the most severe injuries, and **{manu_rate:.0f}%** of them are amputations.
- **Machinery** is the most dangerous source: about **63%** of machinery injuries end in amputation.
- **Falls** cause many hospital visits but almost **no amputations**.
- Last month's injuries predict this month's very well (**R² = 0.96**).
- **{best['Model']}** predicts amputations with **{best['Accuracy']*100:.1f}%** accuracy.
""")

# ==================================================================
# EXPLORE
# ==================================================================
elif page == "📊 Explore the Data":
    st.title("📊 Explore the Data")
    st.write("Use the filters to look at any industry or time period. All charts update automatically.")

    f1, f2 = st.columns([1, 2])
    years = f1.slider("Years", 2015, 2025, (2015, 2025))
    all_sectors = sorted(data['Sector'].unique())
    sectors = f2.multiselect("Industry sectors (leave empty for all)", all_sectors)
    df = data[(data['Year'] >= years[0]) & (data['Year'] <= years[1])]
    if sectors:
        df = df[df['Sector'].isin(sectors)]

    if df.empty:
        st.warning("No records match these filters. Try a wider range.")
        st.stop()

    c1, c2, c3 = st.columns(3)
    with c1: card("Injuries in selection", f"{len(df):,}")
    with c2: card("Amputations", f"{df['Amputation_Flag'].sum():,}", f"{df['Amputation_Flag'].mean()*100:.1f}% of injuries")
    with c3: card("Most common accident", df['Event_Type'].value_counts().index[0])
    st.write("")

    tab1, tab2, tab3, tab4 = st.tabs(["Over time", "By industry", "Causes", "Body parts"])

    with tab1:
        yearly = df.groupby('Year').size().reset_index(name='Injuries')
        fig = px.bar(yearly, x='Year', y='Injuries', color_discrete_sequence=[NAVY], text='Injuries')
        fig.update_traces(texttemplate='%{text:,}', textposition='outside')
        fig.update_layout(title="Severe injuries per year", height=420, yaxis_title="Number of injuries")
        st.plotly_chart(fig, width="stretch")
        explain("Injuries peaked in 2018 and dropped in 2020, most likely because of COVID-19 lockdowns. 2025 includes data only up to November.")

    with tab2:
        a, b = st.columns(2)
        sec = df['Sector'].value_counts().sort_values().reset_index()
        sec.columns = ['Sector', 'Injuries']
        fig = px.bar(sec, x='Injuries', y='Sector', orientation='h', color_discrete_sequence=[NAVY])
        fig.update_layout(title="Number of injuries by sector", height=450, yaxis_title="")
        a.plotly_chart(fig, width="stretch")
        rate = (df.groupby('Sector')['Amputation_Flag'].mean() * 100).sort_values().reset_index()
        rate.columns = ['Sector', 'Amputation %']
        fig = px.bar(rate, x='Amputation %', y='Sector', orientation='h', color='Amputation %',
                     color_continuous_scale=['#FFE8CC', ORANGE, RED])
        fig.update_layout(title="% of injuries that are amputations", height=450, yaxis_title="", coloraxis_showscale=False)
        b.plotly_chart(fig, width="stretch")
        explain("The left chart shows <b>how many</b> injuries each sector has; the right chart shows <b>how serious</b> they are. Construction has many injuries but few amputations, while Manufacturing is high on both.")

    with tab3:
        a, b = st.columns(2)
        src = (df.groupby('Source_Type')['Amputation_Flag'].mean() * 100).sort_values().reset_index()
        src.columns = ['Source', 'Amputation %']
        fig = px.bar(src, x='Amputation %', y='Source', orientation='h', color='Amputation %',
                     color_continuous_scale=['#FFE8CC', ORANGE, RED])
        fig.update_layout(title="Amputation rate by source of injury", height=420, yaxis_title="", coloraxis_showscale=False)
        a.plotly_chart(fig, width="stretch")
        ev = df['Event_Type'].value_counts().sort_values().reset_index()
        ev.columns = ['Accident type', 'Injuries']
        fig = px.bar(ev, x='Injuries', y='Accident type', orientation='h', color_discrete_sequence=[NAVY])
        fig.update_layout(title="Number of injuries by accident type", height=420, yaxis_title="")
        b.plotly_chart(fig, width="stretch")
        explain("Machinery is by far the most dangerous source: most machinery injuries end in an amputation. Machine guards and lockout/tagout are the most important controls.")

    with tab4:
        table = pd.crosstab(df['Event_Type'], df['Body_Part'])
        fig = px.imshow(table, text_auto=True, color_continuous_scale='Reds', aspect='auto')
        fig.update_layout(title="Accident type vs body part injured", height=480, xaxis_title="Body part", yaxis_title="Accident type")
        st.plotly_chart(fig, width="stretch")
        explain("Darker boxes mean more injuries. Most arm and hand injuries come from <b>contact with equipment</b>, while falls mostly hurt the legs, trunk and head.")

    with st.expander("🔬 Is this relationship real? (Chi-square test)"):
        st.write("The chi-square test checks if a feature and amputation are related. If the p-value is below 0.05, they are related.")
        rows = []
        for col in ['Sector', 'Event_Type', 'Source_Type', 'Body_Part']:
            chi2, p, dof, _ = chi2_contingency(pd.crosstab(df[col], df['Amputation_Flag']))
            rows.append([col, round(chi2, 2), f"{p:.4f}", "✅ Related" if p < 0.05 else "❌ Not related"])
        st.dataframe(pd.DataFrame(rows, columns=['Feature', 'Chi-square', 'p-value', 'Result']), hide_index=True, width="stretch")

# ==================================================================
# RISK CHECKER
# ==================================================================
elif page == "🎯 Amputation Risk Checker":
    st.title("🎯 Amputation Risk Checker")
    st.write("Describe an injury situation below. The models will predict whether it is likely to end in an **amputation**.")

    models, sc, columns, results, _ = train_classifiers(data)

    with st.form("risk_form"):
        a, b = st.columns(2)
        sector = a.selectbox("🏭 Industry sector", sorted([s for s in data['Sector'].unique() if s != 'Other']) + ['Other'],
                             index=sorted(data['Sector'].unique()).index('Manufacturing'))
        event = b.selectbox("⚠️ What happened?", list(EVENT.values()) + ['Other'], index=5)
        source = a.selectbox("⚙️ What caused it?", list(SOURCE.values()) + ['Other'], index=2)
        body = b.selectbox("✋ Body part", list(BODY.values()) + ['Other'], index=3)
        month = a.selectbox("📅 Month", MONTHS, index=0)
        submitted = st.form_submit_button("Check risk", type="primary", width="stretch")

    if submitted:
        row = pd.DataFrame([[sector, event, source, body, MONTHS.index(month) + 1]], columns=FEATURES)
        x = pd.get_dummies(row, columns=FEATURES[:4]).reindex(columns=columns, fill_value=0)
        x = sc.transform(x)
        prob = models["Logistic Regression"].predict_proba(x)[0, 1]
        votes = {name: int(m.predict(x)[0]) for name, m in models.items()}
        yes = sum(votes.values())

        # how often this exact situation ended in amputation in the real data
        same = data[(data['Sector'] == sector) & (data['Event_Type'] == event) &
                    (data['Source_Type'] == source) & (data['Body_Part'] == body)]

        if prob >= 0.6:
            colour, level, msg = RED, "HIGH RISK", "This kind of injury often ends in an amputation."
        elif prob >= 0.3:
            colour, level, msg = AMBER, "MEDIUM RISK", "Some injuries like this end in an amputation."
        else:
            colour, level, msg = GREEN, "LOW RISK", "Injuries like this rarely end in an amputation."

        st.write("")
        left, right = st.columns([1, 1.2])
        with left:
            st.markdown(f'<div class="risk" style="background:{colour}"><div class="small">{level}</div>'
                        f'<div class="big">{prob*100:.0f}%</div><div class="small">chance of amputation</div></div>',
                        unsafe_allow_html=True)
            st.caption(msg + " (Probability from Logistic Regression.)")
        with right:
            st.markdown(f"**What do all 5 models say?** {yes} of 5 predict an amputation.")
            vt = pd.DataFrame({"Model": list(votes.keys()),
                               "Prediction": ["🔴 Amputation" if v else "🟢 No amputation" for v in votes.values()],
                               "Model accuracy": [f"{results.set_index('Model').loc[n, 'Accuracy']*100:.1f}%" for n in votes]})
            st.dataframe(vt, hide_index=True, width="stretch")
            if len(same) >= 10:
                st.markdown(f"📁 In the real data, **{len(same):,}** injuries match this situation, "
                            f"and **{same['Amputation_Flag'].mean()*100:.0f}%** of them were amputations.")
            else:
                st.markdown("📁 There are very few real injuries with exactly this combination, so treat the result with care.")

        st.subheader("🛡️ Safety tips for this situation")
        tips = {
            'Machinery': "Keep machine guards in place and never reach past them while the machine is running.",
            'Tools & Equipment': "Use the right tool for the job, keep blade guards on, and wear cut-resistant gloves.",
            'Vehicles': "Separate people from vehicles with marked walkways, and use spotters when reversing.",
            'Parts & Materials': "Secure loads, use lifting aids, and keep hands clear of pinch points when moving materials.",
            'Containers': "Check containers before lifting and use trolleys or lifting equipment for heavy loads.",
            'Floors & Surfaces': "Keep floors clean and dry, and report spills or damaged surfaces straight away.",
            'Chemicals': "Read the safety data sheet and wear the right gloves, goggles and protective clothing.",
        }
        event_tips = {
            'Contact with Equipment': "Apply lockout/tagout before cleaning, clearing jams or doing maintenance.",
            'Fall': "Use fall protection at height and check ladders and scaffolding before use.",
            'Transport': "Wear seat belts and follow site speed limits.",
            'Fire/Explosion': "Control ignition sources and store flammable materials correctly.",
            'Overexertion': "Use lifting aids and team lifts, and take regular breaks.",
            'Harmful Exposure': "Use ventilation and PPE, and limit time in hot or harmful areas.",
            'Violence': "Report threats early and follow workplace violence procedures.",
        }
        shown = [t for t in [tips.get(source), event_tips.get(event)] if t]
        if body == 'Arm/Hand':
            shown.append("Hands are the most at-risk body part: wear suitable gloves and never bypass safety interlocks.")
        shown.append("Make sure everyone is trained on the task and knows how to report near-misses.")
        for t in shown:
            st.markdown(f'<div class="tip">✔️ {t}</div>', unsafe_allow_html=True)
    else:
        st.info("👆 Choose the options and press **Check risk** to see the result.")

# ==================================================================
# FORECAST
# ==================================================================
elif page == "📈 Monthly Injury Forecast":
    st.title("📈 Monthly Injury Forecast")
    st.write("Pick an industry to see its monthly injuries and the **predicted number for next month**, using Linear Regression.")

    lr, scaler, le, metrics, _, _ = train_regression(data)
    sector = st.selectbox("🏭 Industry sector", sorted(data['Sector'].unique()),
                          index=sorted(data['Sector'].unique()).index('Manufacturing'))

    series = data[data['Sector'] == sector].groupby(['Year', 'Month']).size().reset_index(name='Injuries')
    series['Date'] = pd.to_datetime(dict(year=series['Year'], month=series['Month'], day=1))
    last = series.iloc[-1]
    next_date = last['Date'] + pd.DateOffset(months=1)
    x = pd.DataFrame([[last['Injuries'], next_date.month, le.transform([sector])[0]]],
                     columns=['Last_Month', 'Month', 'Sector_Code'])
    forecast = max(0, lr.predict(scaler.transform(x))[0])

    c1, c2, c3 = st.columns(3)
    with c1: card(f"Injuries in {last['Date']:%b %Y}", f"{int(last['Injuries'])}", "last month in the data")
    with c2: card(f"Forecast for {next_date:%b %Y}", f"{forecast:.0f}", f"± {metrics['MAE']:.0f} injuries (average error)")
    with c3: card("Model accuracy (R²)", f"{metrics['R2']:.3f}", "1.0 would be perfect")
    st.write("")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=series['Date'], y=series['Injuries'], mode='lines', name='Actual injuries',
                             line=dict(color=NAVY, width=2)))
    fig.add_trace(go.Scatter(x=[last['Date'], next_date], y=[last['Injuries'], forecast], mode='lines+markers',
                             name='Forecast', line=dict(color=ORANGE, width=3, dash='dash'), marker=dict(size=10)))
    fig.update_layout(title=f"{sector}: injuries per month", height=450, yaxis_title="Injuries",
                      legend=dict(orientation='h', y=1.08, x=0))
    st.plotly_chart(fig, width="stretch")
    explain("The model mainly uses <b>last month's injuries</b> to predict next month (correlation 0.98). "
            "This helps safety teams plan inspections and resources in advance.")

# ==================================================================
# MODEL RESULTS
# ==================================================================
elif page == "🏆 Model Results":
    st.title("🏆 Model Results")
    models, sc, columns, results, matrices = train_classifiers(data)
    _, _, _, metrics, y_test, y_pred = train_regression(data)

    tab1, tab2 = st.tabs(["Classification (amputation or not)", "Regression (monthly injuries)"])
    with tab1:
        best = results.sort_values('F1 Score', ascending=False).iloc[0]['Model']
        st.success(f"🏅 Best model: **{best}** (highest accuracy and F1 score)")
        show = results.copy()
        for c in ['Accuracy', 'Precision', 'Recall', 'F1 Score']:
            show[c] = (show[c] * 100).round(1)
        st.dataframe(show.sort_values('F1 Score', ascending=False), hide_index=True, width="stretch",
                     column_config={c: st.column_config.ProgressColumn(c, format="%.1f%%", min_value=0, max_value=100)
                                    for c in ['Accuracy', 'Precision', 'Recall', 'F1 Score']})
        long = results.melt(id_vars='Model', var_name='Metric', value_name='Score')
        fig = px.bar(long, x='Model', y='Score', color='Metric', barmode='group',
                     color_discrete_sequence=[NAVY, '#4C8BC4', ORANGE, GREY])
        fig.update_layout(title="Comparison of classification models", height=430, yaxis_range=[0, 1])
        st.plotly_chart(fig, width="stretch")

        st.subheader("Confusion matrix")
        chosen = st.selectbox("Choose a model", list(matrices.keys()), index=list(matrices.keys()).index(best))
        cm = matrices[chosen]
        a, b = st.columns([1, 1])
        fig = px.imshow(cm, text_auto=True, color_continuous_scale='Blues',
                        x=['No amputation', 'Amputation'], y=['No amputation', 'Amputation'])
        fig.update_layout(height=380, xaxis_title="Predicted", yaxis_title="Actual", coloraxis_showscale=False)
        a.plotly_chart(fig, width="stretch")
        b.markdown(f"""
**How to read it** (5,000 test records):
- ✅ **{cm[1,1]:,}** amputations correctly found
- ✅ **{cm[0,0]:,}** non-amputations correctly found
- ⚠️ **{cm[0,1]:,}** false alarms
- ❌ **{cm[1,0]:,}** amputations missed

In safety work, **missing a real amputation is worse than a false alarm**, so a low number of misses is important.
""")
        explain("Accuracy = % of all predictions that are correct. Precision = of the predicted amputations, how many were real. "
                "Recall = of the real amputations, how many were found. F1 = balance of precision and recall.")

    with tab2:
        c1, c2, c3, c4 = st.columns(4)
        with c1: card("MAE", f"{metrics['MAE']:.2f}", "average error")
        with c2: card("MSE", f"{metrics['MSE']:.2f}")
        with c3: card("RMSE", f"{metrics['RMSE']:.2f}")
        with c4: card("R²", f"{metrics['R2']:.3f}", "explains 96% of variation")
        st.write("")
        fig = px.scatter(x=y_test, y=y_pred, opacity=0.6, color_discrete_sequence=[NAVY],
                         labels={'x': 'Actual injuries', 'y': 'Predicted injuries'})
        lo, hi = float(min(y_test)), float(max(y_test))
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode='lines', name='Perfect prediction', line=dict(color=RED)))
        fig.update_layout(title="Actual vs predicted monthly injuries", height=450)
        st.plotly_chart(fig, width="stretch")
        explain("Each dot is one sector-month from the test data. The closer the dots are to the red line, the better the prediction.")

# ==================================================================
# STATE GROUPS
# ==================================================================
elif page == "🗺️ State Groups":
    st.title("🗺️ State Groups (K-Means Clustering)")
    st.write("States are grouped by their number of injuries, amputation rate and how much of their injuries come from manufacturing.")
    states, wcss = train_clusters(data)
    colours = {"Low manufacturing": GREEN, "Manufacturing-heavy": RED, "Very high volume": NAVY}

    pick = st.selectbox("🔎 Find a state", sorted(states.index), index=sorted(states.index).index('Ohio'))
    s = states.loc[pick]
    c1, c2, c3, c4 = st.columns(4)
    with c1: card("Group", s['Group'])
    with c2: card("Injuries", f"{int(s['Injuries']):,}")
    with c3: card("Amputation rate", f"{s['Amputation_Rate']*100:.1f}%")
    with c4: card("From manufacturing", f"{s['Manufacturing_Share']*100:.0f}%")
    st.write("")

    plot = states.reset_index()
    fig = px.scatter(plot, x='Manufacturing_Share', y='Amputation_Rate', color='Group', size='Injuries',
                     hover_name='State', color_discrete_map=colours, size_max=45,
                     labels={'Manufacturing_Share': 'Share of injuries from manufacturing', 'Amputation_Rate': 'Amputation rate'})
    fig.update_layout(height=500, xaxis_tickformat='.0%', yaxis_tickformat='.0%', legend=dict(orientation='h', y=1.08, x=0))
    st.plotly_chart(fig, width="stretch")
    explain("Each bubble is a state (bigger bubble = more injuries). States with more manufacturing (to the right) have a higher amputation rate.")

    summary = states.groupby('Group').agg(States=('Injuries', 'size'), Avg_Injuries=('Injuries', 'mean'),
                                          Amputation_Rate=('Amputation_Rate', 'mean'),
                                          Manufacturing_Share=('Manufacturing_Share', 'mean')).reset_index()
    summary['Avg_Injuries'] = summary['Avg_Injuries'].round(0).astype(int)
    summary['Amputation_Rate'] = (summary['Amputation_Rate'] * 100).round(1).astype(str) + '%'
    summary['Manufacturing_Share'] = (summary['Manufacturing_Share'] * 100).round(1).astype(str) + '%'
    st.dataframe(summary, hide_index=True, width="stretch")

    with st.expander("How was the number of groups chosen? (Elbow method)"):
        fig = px.line(x=list(range(1, 9)), y=wcss, markers=True, labels={'x': 'Number of clusters (k)', 'y': 'WCSS'})
        fig.update_traces(line_color=ORANGE)
        fig.update_layout(height=350)
        st.plotly_chart(fig, width="stretch")
        st.write("WCSS drops sharply up to 3–4 clusters and then flattens, so **k = 3** was chosen to keep the groups easy to understand.")

# ==================================================================
# ABOUT
# ==================================================================
else:
    st.title("ℹ️ About this Project")
    st.markdown("""
**Predicting Severe Workplace Injuries using Machine Learning**
INT234 – Predictive Analytics · Lovely Professional University

**Problem:** Thousands of workers are seriously injured at work every year. This project finds where severe injuries happen and predicts which ones lead to an amputation, so that safety teams can act before someone gets hurt.

**Dataset:** [OSHA Severe Injury Reports](https://www.osha.gov/severeinjury), U.S. Department of Labor. It has 105,996 reports from January 2015 to November 2025.

**Steps followed:**
1. Data cleaning: removing columns, handling missing values, creating new columns
2. Encoding (LabelEncoder, one-hot) and scaling (StandardScaler)
3. EDA with charts and the chi-square test
4. Linear Regression to forecast monthly injuries
5. Logistic Regression, KNN, Naive Bayes, Decision Tree and SVM to predict amputation
6. K-Means clustering to group similar states

**Tools:** Python, Pandas, NumPy, Scikit-learn, SciPy, Plotly, Streamlit

**Made by:** Jaskaran Singh · B.Tech CSE (Data Science)
""")
