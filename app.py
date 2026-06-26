import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import ast

# Page configurations
st.set_page_config(
    page_title="Coursera Course Recommendation Engine",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern visual design and theme overrides
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global Typography & Background adjustments */
    html, body, [class*="css"], .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
    }

    /* Main dashboard container styling */
    .stApp {
        background-color: #F5F7FA;
        color: #1F1F1F;
    }

    /* Header styling with gradient text */
    .main-title {
        color: #0056D2;
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
        font-family: 'Outfit', sans-serif;
    }
    
    .sub-title {
        text-align: center;
        color: #6B7280;
        font-size: 1.15rem;
        margin-bottom: 2.5rem;
        font-weight: 400;
    }

    /* Elegant Custom Card Styling */
    .course-card {
        background-color: #FFFFFF;
        border-radius: 12px;
        padding: 1.6rem;
        margin-bottom: 1.2rem;
        border: 1px solid #D1D5DB;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        min-height: 380px;
    }
    
    .course-card:hover {
        transform: translateY(-4px);
        border-color: #0056D2;
        box-shadow: 0 6px 15px rgba(0, 86, 210, 0.15);
    }
    
    /* Highlighted card for Selected course */
    .selected-card {
        background: #EBF3FF;
        border: 2px solid #0056D2;
        box-shadow: 0 4px 15px rgba(0, 86, 210, 0.12);
        min-height: auto;
    }
    
    .card-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #1F1F1F;
        margin-bottom: 0.6rem;
        line-height: 1.35;
        font-family: 'Outfit', sans-serif;
    }
    
    .card-org {
        font-size: 0.95rem;
        color: #0056D2;
        font-weight: 600;
        margin-bottom: 0.8rem;
    }
    
    .card-meta {
        display: flex;
        flex-wrap: wrap;
        gap: 0.6rem;
        margin-bottom: 1.1rem;
    }
    
    .badge {
        padding: 0.25rem 0.65rem;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    
    .badge-rating {
        background-color: rgba(226, 183, 20, 0.12);
        color: #B38C00;
        border: 1px solid rgba(226, 183, 20, 0.25);
    }
    
    .badge-level {
        background-color: rgba(0, 86, 210, 0.08);
        color: #0056D2;
        border: 1px solid rgba(0, 86, 210, 0.2);
    }
    
    .badge-duration {
        background-color: rgba(0, 177, 162, 0.08);
        color: #008277;
        border: 1px solid rgba(0, 177, 162, 0.2);
    }
    
    .card-desc {
        font-size: 0.9rem;
        color: #4B5563;
        line-height: 1.5;
        margin-bottom: 1.2rem;
        flex-grow: 1;
        overflow: hidden;
        text-overflow: ellipsis;
        display: -webkit-box;
        -webkit-line-clamp: 4;
        -webkit-box-orient: vertical;
    }
    
    .skills-container {
        display: flex;
        flex-wrap: wrap;
        gap: 0.4rem;
        margin-bottom: 1.2rem;
    }
    
    .skill-tag {
        font-size: 0.72rem;
        background-color: #F3F4F6;
        color: #4B5563;
        border: 1px solid #D1D5DB;
        padding: 0.15rem 0.55rem;
        border-radius: 100px;
        font-weight: 500;
    }
    
    .card-footer {
        margin-top: auto;
    }
    
    .card-button {
        display: block;
        background: #0056D2;
        color: #FFFFFF !important;
        text-align: center;
        padding: 0.6rem 1rem;
        border-radius: 8px;
        text-decoration: none !important;
        font-weight: 600;
        font-size: 0.9rem;
        transition: all 0.2s ease;
    }
    
    .card-button:hover {
        background: #00419D;
        box-shadow: 0 4px 10px rgba(0, 86, 210, 0.2);
        transform: translateY(-1px);
    }

    /* Score overlay panel in card */
    .metric-row {
        display: flex;
        justify-content: space-between;
        font-size: 0.78rem;
        color: #6B7280;
        margin-top: 0.8rem;
        border-top: 1px solid #E5E7EB;
        padding-top: 0.6rem;
    }
    
    .metric-val {
        color: #1F1F1F;
        font-weight: 600;
    }

    /* Info callout container */
    .info-container {
        background-color: #EBF3FF;
        border-left: 4px solid #0056D2;
        border-radius: 0 8px 8px 0;
        padding: 1rem;
        margin-bottom: 1.5rem;
        color: #1F1F1F;
    }

    /* Streamlit overrides for custom light theme branding */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #D1D5DB !important;
    }
    
    button[data-baseweb="tab"] {
        color: #6B7280 !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 500 !important;
    }
    
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #0056D2 !important;
        font-weight: 700 !important;
        border-bottom-color: #0056D2 !important;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to safely clean fields stored as string representations of lists
def clean_list_column(val):
    if isinstance(val, list):
        return val
    if pd.isna(val):
        return []
    if isinstance(val, str):
        val = val.strip()
        if (val.startswith('[') and val.endswith(']')) or (val.startswith('(') and val.endswith(')')):
            try:
                return ast.literal_eval(val)
            except Exception:
                pass
        return [val]
    return [val]

# Load and preprocess dataset (Cached)
@st.cache_data
def load_data():
    # Load raw cleaned pickle
    df = pd.read_pickle("data/coursera_cleaned.pkl")
    
    # Ensure list types for relevant columns
    list_cols = ['Skill gain', 'Offered By', 'Keyword', 'Modules', 'Instructor']
    for col in list_cols:
        if col in df.columns:
            df[col] = df[col].apply(clean_list_column)
            
    # Calculate Bayesian weighted popularity score
    # m is the minimum reviews required (25th percentile)
    m = df['review_count'].quantile(0.25)
    # C is the global mean rating across the whole dataset
    C = df['Rating'].mean()
    
    df['weighted_score'] = (
        (df['review_count'] / (df['review_count'] + m)) * df['Rating'] +
        (m / (df['review_count'] + m)) * C
    )
    
    # Scale popularity score to [0, 1] range
    scaler = MinMaxScaler()
    df['weighted_score_norm'] = scaler.fit_transform(df[['weighted_score']])
    
    # Prepare indices map
    indices = pd.Series(df.index, index=df['Course Title']).drop_duplicates()
    
    return df, indices, m, C

# Compute TF-IDF Model and Cosine Similarity (Cached as resource)
@st.cache_resource
def compute_similarity(df):
    tfidf = TfidfVectorizer(
        stop_words='english',
        max_features=10000,
        ngram_range=(1,2)
    )
    tfidf_matrix = tfidf.fit_transform(df['content_soup'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return tfidf, tfidf_matrix, cosine_sim

# Load data and similarities
try:
    df, indices, review_threshold, global_mean_rating = load_data()
    tfidf, tfidf_matrix, cosine_sim = compute_similarity(df)
    data_loaded = True
except Exception as e:
    data_loaded = False
    st.error(f"Error loading Coursera dataset: {e}")

# Render visual HTML course cards
def render_course_card(course, selected=False, is_recommendation=False, similarity=None, popularity=None, match_score=None):
    title = course['Course Title']
    rating = course['Rating']
    reviews = course['review_count']
    level = course['Level']
    duration = course['duration_hours']
    url = course['Course Url']
    
    offered_by = ", ".join(course['Offered By']) if course['Offered By'] else "Coursera Partner"
    skills = course['Skill gain']
    desc = course['What you will learn']
    
    if not desc or pd.isna(desc):
        desc = "Discover essential techniques, framework methods, and hands-on case studies in this specialized course curriculum."
        
    desc = desc.replace('"', '&quot;').replace("'", "&#39;")
        
    skills_html = "".join([f'<span class="skill-tag">{s}</span>' for s in skills[:4]])
    selected_class = " selected-card" if selected else ""
    
    metrics_html = ""
    if is_recommendation:
        metrics_html = (
            '<div class="metric-row">'
            f'<span>Cosine Sim: <span class="metric-val">{similarity:.2f}</span></span>'
            f'<span>Popularity: <span class="metric-val">{popularity:.2f}</span></span>'
            f'<span>Match: <span class="metric-val">{match_score:.2f}</span></span>'
            '</div>'
        )
        
    html = (
        f'<div class="course-card{selected_class}">'
        '<div>'
        f'<div class="card-title">{title}</div>'
        f'<div class="card-org">Offered by {offered_by}</div>'
        '<div class="card-meta">'
        f'<span class="badge badge-rating">⭐ {rating:.1f} ({reviews:,} reviews)</span>'
        f'<span class="badge badge-level">📊 {level}</span>'
        f'<span class="badge badge-duration">⏱️ {duration}h</span>'
        '</div>'
        f'<div class="card-desc">{desc}</div>'
        '</div>'
        '<div class="card-footer">'
        f'<div class="skills-container">{skills_html}</div>'
        f'<a class="card-button" href="{url}" target="_blank">View Course ↗</a>'
        f'{metrics_html}'
        '</div>'
        '</div>'
    )
    return html

# Recommendation Engine Logic (Based on Similar Course)
def get_recommendations(df, indices, cosine_sim, course_title, 
                        model_type="Hybrid", top_n=10, alpha=0.7, 
                        selected_levels=None, max_duration=None, 
                        min_rating=None, selected_keywords=None):
    
    if course_title not in indices:
        return None
        
    idx = indices[course_title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    recommendations = []
    for course_idx, sim_score in sim_scores:
        # Exclude the seed course itself
        if course_idx == idx:
            continue
            
        # Skip completely unrelated courses
        if sim_score < 0.01:
            continue
            
        popularity_score = df.loc[course_idx, 'weighted_score_norm']
        
        # Calculate score based on recommendation method
        if model_type == "Hybrid":
            score = alpha * sim_score + (1 - alpha) * popularity_score
        elif model_type == "TF-IDF (Content-Based)":
            score = sim_score
        else:  # Popularity-Based
            score = popularity_score
            
        course_row = df.iloc[course_idx]
        
        # Filter 1: Difficulty Level
        if selected_levels and course_row['Level'] not in selected_levels:
            continue
            
        # Filter 2: Duration Limit
        if max_duration and course_row['duration_hours'] > max_duration:
            continue
            
        # Filter 3: Minimum Rating
        if min_rating and course_row['Rating'] < min_rating:
            continue
            
        # Filter 4: Keyword categories
        if selected_keywords:
            course_kws = course_row['Keyword']
            if not any(kw in selected_keywords for kw in course_kws):
                continue
                
        recommendations.append({
            'index': course_idx,
            'score': score,
            'similarity': sim_score,
            'popularity': popularity_score,
            'row_data': course_row.to_dict()
        })
        
    # Sort by recommendation rank
    recommendations = sorted(recommendations, key=lambda x: x['score'], reverse=True)
    recommendations = recommendations[:top_n]
    
    results = []
    for r in recommendations:
        c_data = r['row_data']
        c_data['Match Score'] = r['score']
        c_data['Similarity'] = r['similarity']
        c_data['Popularity'] = r['popularity']
        results.append(c_data)
        
    return pd.DataFrame(results)

# Recommendation Engine Logic (Based on Free Text Search Query)
def get_recommendations_for_query(df, tfidf, tfidf_matrix, query,
                                  model_type="Hybrid", top_n=10, alpha=0.7,
                                  selected_levels=None, max_duration=None,
                                  min_rating=None, selected_keywords=None):
    if not query.strip():
        return pd.DataFrame()
        
    # Transform query to TF-IDF vector space
    query_vec = tfidf.transform([query])
    # Compute similarity between query vector and all course vectors
    sim_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    
    recommendations = []
    for course_idx, sim_score in enumerate(sim_scores):
        # Skip completely unrelated courses (strict threshold for queries)
        if sim_score < 0.02:
            continue
            
        popularity_score = df.loc[course_idx, 'weighted_score_norm']
        
        if model_type == "Hybrid":
            score = alpha * sim_score + (1 - alpha) * popularity_score
        elif model_type == "TF-IDF (Content-Based)":
            score = sim_score
        else: # Popularity-Based
            score = popularity_score
            
        course_row = df.iloc[course_idx]
        
        # Apply filters
        if selected_levels and course_row['Level'] not in selected_levels:
            continue
        if max_duration and course_row['duration_hours'] > max_duration:
            continue
        if min_rating and course_row['Rating'] < min_rating:
            continue
        if selected_keywords:
            course_kws = course_row['Keyword']
            if not any(kw in selected_keywords for kw in course_kws):
                continue
                
        recommendations.append({
            'index': course_idx,
            'score': score,
            'similarity': sim_score,
            'popularity': popularity_score,
            'row_data': course_row.to_dict()
        })
        
    # Sort by score descending
    recommendations = sorted(recommendations, key=lambda x: x['score'], reverse=True)
    recommendations = recommendations[:top_n]
    
    results = []
    for r in recommendations:
        c_data = r['row_data']
        c_data['Match Score'] = r['score']
        c_data['Similarity'] = r['similarity']
        c_data['Popularity'] = r['popularity']
        results.append(c_data)
        
    return pd.DataFrame(results)

# Catalog Browser Logic
def browse_catalog(df, selected_levels=None, max_duration=None, min_rating=None, selected_keywords=None, sort_by="Rating (High to Low)", top_n=20):
    filtered_df = df.copy()
    
    if selected_levels:
        filtered_df = filtered_df[filtered_df['Level'].isin(selected_levels)]
        
    if max_duration:
        filtered_df = filtered_df[filtered_df['duration_hours'] <= max_duration]
        
    if min_rating:
        filtered_df = filtered_df[filtered_df['Rating'] >= min_rating]
        
    if selected_keywords:
        filtered_df = filtered_df[filtered_df['Keyword'].apply(
            lambda kws: any(kw in selected_keywords for kw in kws)
        )]
        
    # Sort selections
    if sort_by == "Rating (High to Low)":
        filtered_df = filtered_df.sort_values(by='Rating', ascending=False)
    elif sort_by == "Popularity (Bayesian)":
        filtered_df = filtered_df.sort_values(by='weighted_score', ascending=False)
    elif sort_by == "Duration (Short to Long)":
        filtered_df = filtered_df.sort_values(by='duration_hours', ascending=True)
    elif sort_by == "Reviews (Most to Least)":
        filtered_df = filtered_df.sort_values(by='review_count', ascending=False)
        
    return filtered_df.head(top_n)

# Header Section
st.markdown('<div class="main-title">Coursera Course Recommender</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Discover top-rated courses matching your target topics and difficulty level</div>', unsafe_allow_html=True)

if data_loaded:
    # Get categorical filters for widgets
    all_levels = sorted(df['Level'].dropna().unique())
    all_keywords = sorted(list(set(kw for kws in df['Keyword'] for kw in kws)))
    
    st.sidebar.markdown('<div style="font-size: 1.6rem; font-weight: 800; color: #0056D2; margin-top: -1.5rem; margin-bottom: 1.5rem; font-family: \'Outfit\', sans-serif;">Coursera Portal</div>', unsafe_allow_html=True)
    st.sidebar.markdown("### ⚙️ Recommendation Settings")
    
    model_type = st.sidebar.selectbox(
        "Recommendation Model",
        ["Hybrid", "TF-IDF (Content-Based)", "Popularity-Based"],
        help="Hybrid balances text similarity and course rating/popularity. TF-IDF uses pure description overlap. Popularity lists globally top-rated courses."
    )
    
    alpha = 0.7
    if model_type == "Hybrid":
        alpha = st.sidebar.slider(
            "Hybrid Weight (Alpha)",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.05,
            help="Higher values prioritize keyword similarity; lower values prioritize popularity and rating."
        )
        
    top_n = st.sidebar.slider("Number of Recommendations", 2, 20, 8, step=2)
    
    st.sidebar.markdown("### 🔍 Filter Criteria")
    
    selected_levels = st.sidebar.multiselect(
        "Difficulty Levels",
        options=all_levels,
        default=all_levels
    )
    
    max_duration = st.sidebar.slider(
        "Max Duration (Hours)",
        min_value=1,
        max_value=150,
        value=150,
        step=5
    )
    
    min_rating = st.sidebar.slider(
        "Minimum Rating",
        min_value=3.0,
        max_value=5.0,
        value=4.0,
        step=0.1
    )
    
    selected_keywords = st.sidebar.multiselect(
        "Filter by Subject Categories",
        options=all_keywords
    )
    
    # Main Dashboard Tabs
    tab_recommender, tab_insights = st.tabs([
        "🎯 Personal Recommender", 
        "📊 Dataset Insights"
    ])
    
    # Tab 1: Personal Recommender
    with tab_recommender:
        st.markdown('<div class="info-container">💡 <strong>How it works:</strong> Type what you want to learn (e.g., "gen ai") or select a course you already liked to find the best matching recommendations.</div>', unsafe_allow_html=True)
        
        # Search Mode Selection
        search_mode = st.radio(
            "Choose Search Mode:",
            ["🔍 Search by Topic / Keywords (Free Text)", "🎓 Search Similar to an Existing Course"],
            horizontal=True
        )
        
        st.markdown("---")
        
        if search_mode == "🔍 Search by Topic / Keywords (Free Text)":
            query_text = st.text_input(
                "What do you want to learn? (e.g. 'gen ai', 'python', 'deep learning', 'excel'):",
                value="gen ai",
                placeholder="Type topics, tools, or concepts..."
            )
            
            if query_text:
                st.subheader(f"✨ Top Recommendations for: \"{query_text}\"")
                
                recs_df = get_recommendations_for_query(
                    df, tfidf, tfidf_matrix, query_text,
                    model_type=model_type, top_n=top_n, alpha=alpha,
                    selected_levels=selected_levels, max_duration=max_duration,
                    min_rating=min_rating, selected_keywords=selected_keywords
                )
                
                if recs_df is not None and not recs_df.empty:
                    # Grid display: 2 columns of cards
                    grid_cols = st.columns(2)
                    for i, (_, row) in enumerate(recs_df.iterrows()):
                        col_pos = i % 2
                        with grid_cols[col_pos]:
                            st.html(
                                render_course_card(
                                    row,
                                    is_recommendation=True,
                                    similarity=row['Similarity'],
                                    popularity=row['Popularity'],
                                    match_score=row['Match Score']
                                )
                            )
                else:
                    st.warning("No recommendations found matching the select sidebar filters. Try expanding your search options (e.g. increase max duration, add more difficulty levels, or select fewer categories).")
                    
        else:
            # Course Autocomplete selectbox
            course_list = df['Course Title'].tolist()
            # Default selectbox to a popular and interesting data science course
            default_idx = course_list.index("AI Capstone Project with Deep Learning") if "AI Capstone Project with Deep Learning" in course_list else 0
            
            selected_course_title = st.selectbox(
                "Select a course you enjoyed:",
                options=course_list,
                index=default_idx
            )
            
            if selected_course_title:
                st.subheader("🔍 Selected Course")
                selected_course_row = df.loc[indices[selected_course_title]].iloc[0] if isinstance(df.loc[indices[selected_course_title]], pd.DataFrame) else df.loc[indices[selected_course_title]]
                
                # Render Selected course details (Full width)
                st.html(render_course_card(selected_course_row, selected=True))
                
                # Expandable details
                with st.expander("More Course Info"):
                    st.write(f"**Instructors:** {', '.join(selected_course_row['Instructor'])}")
                    st.write(f"**Keywords:** {', '.join(selected_course_row['Keyword'])}")
                    if selected_course_row['Modules']:
                        st.write("**Syllabus / Modules:**")
                        for mod in selected_course_row['Modules'][:6]:
                            st.write(f"- {mod}")
                            
                st.markdown("---")
                st.subheader(f"✨ Recommended Courses")
                
                recs_df = get_recommendations(
                    df, indices, cosine_sim, selected_course_title,
                    model_type=model_type, top_n=top_n, alpha=alpha,
                    selected_levels=selected_levels, max_duration=max_duration,
                    min_rating=min_rating, selected_keywords=selected_keywords
                )
                
                if recs_df is not None and not recs_df.empty:
                    # Grid display: 2 columns of cards
                    grid_cols = st.columns(2)
                    for i, (_, row) in enumerate(recs_df.iterrows()):
                        col_pos = i % 2
                        with grid_cols[col_pos]:
                            st.html(
                                render_course_card(
                                    row,
                                    is_recommendation=True,
                                    similarity=row['Similarity'],
                                    popularity=row['Popularity'],
                                    match_score=row['Match Score']
                                )
                            )
                else:
                    st.warning("No recommendations found matching the select sidebar filters. Try expanding your search options (e.g. increase max duration, add more difficulty levels, or select fewer categories).")
                    
    # Tab 2: Dataset Insights & Statistics
    with tab_insights:
        st.subheader("📊 Dataset Statistics & Learning Path Insights")
        st.markdown("Get a bird's eye view of the 6,400+ Coursera courses catalogued in this project.")
        
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            # Difficulty level distribution
            level_counts = df['Level'].value_counts().reset_index()
            level_counts.columns = ['Level', 'Count']
            fig1 = px.pie(
                level_counts, 
                values='Count', 
                names='Level', 
                title='Distribution of Course Difficulty Levels',
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Safe,
                template="plotly_white"
            )
            fig1.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#1F1F1F')
            )
            st.plotly_chart(fig1, use_container_width=True)
            
        with col_chart2:
            # Rating histogram
            fig2 = px.histogram(
                df,
                x='Rating',
                nbins=15,
                title='Course Rating Distribution',
                color_discrete_sequence=['#0056D2'],
                template="plotly_white"
            )
            fig2.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#1F1F1F'),
                xaxis_title='Rating',
                yaxis_title='Number of Courses'
            )
            st.plotly_chart(fig2, use_container_width=True)
            
        st.markdown("---")
        col_chart3, col_chart4 = st.columns(2)
        
        with col_chart3:
            # Top institutions
            inst_df = df.explode('Offered By')
            top_inst = inst_df['Offered By'].value_counts().head(12).reset_index()
            top_inst.columns = ['Institution', 'Course Count']
            fig3 = px.bar(
                top_inst, 
                x='Course Count', 
                y='Institution', 
                orientation='h', 
                title='Top 12 Course Providers/Partners',
                color='Course Count',
                color_continuous_scale='Blues',
                template="plotly_white"
            )
            fig3.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#1F1F1F'),
                yaxis=dict(categoryorder='total ascending'),
                coloraxis_showscale=False
            )
            st.plotly_chart(fig3, use_container_width=True)
            
        with col_chart4:
            # Course Duration Boxplot
            fig4 = px.box(
                df,
                x='Level',
                y='duration_hours',
                color='Level',
                title='Distribution of Course Completion Times (Hours)',
                color_discrete_sequence=px.colors.qualitative.Safe,
                template="plotly_white"
            )
            fig4.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#1F1F1F'),
                xaxis_title='Difficulty Level',
                yaxis_title='Duration (Hours)'
            )
            st.plotly_chart(fig4, use_container_width=True)
else:
    st.info("Ensure the dataset file 'data/coursera_cleaned.pkl' is present in the project directory.")
