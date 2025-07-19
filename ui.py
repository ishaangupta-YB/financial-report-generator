import streamlit as st
import plotly.express as px
import time


def apply_custom_css():
    """Apply custom CSS for beautiful styling."""
    st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }

    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }

    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }

    .feature-card {
        background: rgba(255, 255, 255, 0.95);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }

    .report-section {
        background: rgba(255, 255, 255, 0.98);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        border-left: 5px solid #667eea;
    }

    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem;
    }

    .success-banner {
        background: linear-gradient(90deg, #56ab2f 0%, #a8e6cf 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }

    .error-banner {
        background: linear-gradient(90deg, #ff416c 0%, #ff4b2b 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }

    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }

    .sidebar .stSelectbox > div > div {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }

    h1, h2, h3 {
        color: #2c3e50;
    }

    .report-text {
        font-size: 1.1rem;
        line-height: 1.6;
        color: #34495e;
        text-align: justify;
    }

    .table-container {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)


def create_header():
    """Create the main application header."""
    st.markdown("""
    <div class="main-header">
        <h1>🏦 Financial Report Generator</h1>
        <p style="font-size: 1.2rem; margin-bottom: 0;">
            AI-Powered Financial Analysis & Report Generation
        </p>
    </div>
    """, unsafe_allow_html=True)


def create_sidebar():
    """Create the sidebar with configuration options."""
    with st.sidebar:
        st.markdown("### 🛠️ Configuration")

        # API Keys section
        st.markdown("#### API Keys")
        openai_key = st.text_input(
            "OpenAI API Key",
            type="password",
            placeholder="sk-...",
            help="Enter your OpenAI API key"
        )

        llama_cloud_key = st.text_input(
            "LlamaCloud API Key",
            type="password",
            placeholder="llx-...",
            help="Enter your LlamaCloud API key"
        )

        # Index configuration
        st.markdown("#### Index Configuration")
        index_name = st.text_input(
            "Index Name",
            value="apple_tesla_demo_2",
            help="Name of your LlamaCloud index"
        )

        project_name = st.text_input(
            "Project Name",
            value="llamacloud_demo",
            help="Name of your LlamaCloud project"
        )

        # Example queries
        st.markdown("#### 📝 Example Queries")
        example_queries = [
            "Compare Tesla and Apple's assets and liabilities for 2021",
            "Analyze Apple's gross margin trends from 2020-2023",
            "Provide a summary of Tesla's 2023 performance",
            "Compare revenue growth between Apple and Tesla",
            "Analyze the cash flow patterns of both companies"
        ]

        selected_example = st.selectbox(
            "Choose an example query:",
            [""] + example_queries,
            help="Select a pre-written query to try"
        )

        if st.button("🔄 Use Example Query", help="Load the selected example"):
            if selected_example:
                st.session_state.example_query = selected_example

        return openai_key, llama_cloud_key, index_name, project_name, selected_example


def create_status_indicators():
    """Create status indicators for the application."""
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.session_state.get('initialized', False):
            st.markdown("""
            <div class="metric-card">
                <h4>✅ Connected</h4>
                <p>Ready to generate</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="metric-card" style="background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%);">
                <h4>❌ Not Connected</h4>
                <p>Enter API keys</p>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        reports_generated = st.session_state.get('reports_generated', 0)
        st.markdown(f"""
        <div class="metric-card">
            <h4>📊 Reports</h4>
            <p>{reports_generated} Generated</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        avg_time = st.session_state.get('avg_generation_time', 0)
        st.markdown(f"""
        <div class="metric-card">
            <h4>⏱️ Avg Time</h4>
            <p>{avg_time:.1f}s</p>
        </div>
        """, unsafe_allow_html=True)


def create_query_interface():
    """Create the main query interface."""
    st.markdown("""
    <div class="feature-card">
        <h3>💭 Ask Your Financial Question</h3>
        <p>Enter your query about financial data, comparisons, or analysis below.</p>
    </div>
    """, unsafe_allow_html=True)

    # Use example query if available
    default_query = st.session_state.get('example_query', '')

    query = st.text_area(
        "Enter your financial analysis query:",
        value=default_query,
        height=100,
        placeholder="e.g., Compare the revenue growth of Apple and Tesla over the last 3 years...",
        help="Describe what financial analysis or comparison you'd like to see"
    )

    # Clear the example query after use
    if 'example_query' in st.session_state:
        del st.session_state.example_query

    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        generate_button = st.button(
            "🚀 Generate Report",
            disabled=not st.session_state.get('initialized', False),
            help="Generate a comprehensive financial report based on your query"
        )

    return query, generate_button


def display_loading_animation():
    """Display a loading animation during report generation."""
    with st.spinner(""):
        progress_bar = st.progress(0)
        status_text = st.empty()

        steps = [
            "🔍 Analyzing your query...",
            "📚 Searching financial documents...",
            "🧠 Processing information...",
            "📊 Generating insights...",
            "📝 Formatting report..."
        ]

        for i, step in enumerate(steps):
            status_text.text(step)
            progress_bar.progress((i + 1) / len(steps))
            time.sleep(0.8)

        status_text.empty()
        progress_bar.empty()


def display_report(report_output):
    """Display the generated report with beautiful formatting."""
    st.markdown("""
    <div class="feature-card">
        <h3>📋 Generated Financial Report</h3>
    </div>
    """, unsafe_allow_html=True)

    for i, block in enumerate(report_output.blocks):
        if hasattr(block, 'text'):  # TextBlock
            st.markdown(f"""
            <div class="report-section">
                <div class="report-text">{block.text}</div>
            </div>
            """, unsafe_allow_html=True)

        elif hasattr(block, 'caption'):  # TableBlock
            st.markdown(f"""
            <div class="table-container">
                <h4>{block.caption}</h4>
            </div>
            """, unsafe_allow_html=True)

            # Create DataFrame and display
            df = block.to_df()

            # Style the dataframe
            styled_df = df.style.set_properties(**{
                'background-color': 'white',
                'color': 'black',
                'border-color': '#e0e0e0'
            }).set_table_styles([
                {'selector': 'th',
                 'props': [('background-color', '#667eea'), ('color', 'white'), ('font-weight', 'bold')]},
                {'selector': 'td', 'props': [('border', '1px solid #e0e0e0')]}
            ])

            st.dataframe(styled_df, use_container_width=True)

            # Create a visualization if the data is numeric
            try:
                numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
                if len(numeric_cols) > 0 and len(df) > 1:
                    create_chart_from_table(df, block.caption)
            except:
                pass  # Skip visualization if data isn't suitable


def create_chart_from_table(df, title):
    """Create a chart from table data if possible."""
    try:
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

        if len(numeric_cols) >= 1 and len(df) > 1:
            # Create a bar chart
            if len(df.columns) >= 2:
                fig = px.bar(
                    df,
                    x=df.columns[0],
                    y=numeric_cols[0],
                    title=f"Visualization: {title}",
                    color_discrete_sequence=['#667eea']
                )

                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#2c3e50'),
                    title_font_size=16
                )

                st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        pass  # Silently skip if visualization fails


def display_success_message():
    """Display success message after report generation."""
    st.markdown("""
    <div class="success-banner">
        <h4>✅ Report Generated Successfully!</h4>
        <p>Your financial analysis is ready. You can scroll down to view the complete report.</p>
    </div>
    """, unsafe_allow_html=True)


def display_error_message(error_msg):
    """Display error message."""
    st.markdown(f"""
    <div class="error-banner">
        <h4>❌ Error Occurred</h4>
        <p>{error_msg}</p>
    </div>
    """, unsafe_allow_html=True)


def create_footer():
    """Create application footer."""
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: white; padding: 2rem;">
        <p>🤖 Powered by LlamaIndex, OpenAI, and Streamlit</p>
        <p style="font-size: 0.9rem;">Built for intelligent financial analysis and reporting</p>
    </div>
    """, unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables."""
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
    if 'generator' not in st.session_state:
        st.session_state.generator = None
    if 'reports_generated' not in st.session_state:
        st.session_state.reports_generated = 0
    if 'avg_generation_time' not in st.session_state:
        st.session_state.avg_generation_time = 0.0
    if 'generation_times' not in st.session_state:
        st.session_state.generation_times = []