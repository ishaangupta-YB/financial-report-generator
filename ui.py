import streamlit as st
import pandas as pd
import plotly.express as px
import time


def apply_custom_css():
    """Apply simple, minimal CSS styling with good contrast."""
    st.markdown("""
    <style>
    /* Clean, minimal design with good contrast */
    .stApp {
        background-color: #ffffff;
        color: #1a1a1a;
    }
    
    /* Main content styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1000px;
    }
    
    /* Header styling */
    .app-header {
        background: linear-gradient(90deg, #2563eb 0%, #1d4ed8 100%);
        color: white;
        padding: 2rem;
        border-radius: 8px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }
    
    .app-title {
        font-size: 2rem;
        font-weight: 600;
        margin: 0;
    }
    
    .app-subtitle {
        font-size: 1rem;
        opacity: 0.9;
        margin-top: 0.5rem;
    }
    
    /* Status cards */
    .status-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1.5rem 0;
    }
    
    .status-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1.5rem;
        text-align: center;
    }
    
    .status-card.connected {
        background: #f0fdf4;
        border-color: #bbf7d0;
        color: #166534;
    }
    
    .status-card.disconnected {
        background: #fef2f2;
        border-color: #fecaca;
        color: #dc2626;
    }
    
    .status-number {
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .status-label {
        font-size: 0.875rem;
        opacity: 0.8;
    }
    
    /* Card styling */
    .card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    
    .card-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: #374151;
        margin-bottom: 1rem;
        border-bottom: 1px solid #e5e7eb;
        padding-bottom: 0.5rem;
    }
    
    /* Button styling */
    .stButton > button {
        background: #2563eb;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: background-color 0.2s;
    }
    
    .stButton > button:hover {
        background: #1d4ed8;
    }
    
    .stButton > button:disabled {
        background: #9ca3af;
        color: #ffffff;
    }
    
    /* Form styling */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stSelectbox > div > div > select {
        border: 1px solid #d1d5db;
        border-radius: 6px;
        background: #ffffff;
    }
    
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: #2563eb;
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8fafc;
    }
    
    /* Loading spinner */
    .loading-container {
        text-align: center;
        padding: 2rem;
        background: #f8fafc;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .loading-text {
        color: #6b7280;
        margin-top: 1rem;
    }
    
    /* Alert messages */
    .alert {
        padding: 1rem;
        border-radius: 6px;
        margin: 1rem 0;
    }
    
    .alert-success {
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        color: #166534;
    }
    
    .alert-error {
        background: #fef2f2;
        border: 1px solid #fecaca;
        color: #dc2626;
    }
    
    .alert-info {
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        color: #1d4ed8;
    }
    
    /* Report styling */
    .report-section {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    
    .report-text {
        line-height: 1.6;
        color: #374151;
    }
    
    /* Progress bar */
    .stProgress > div > div > div {
        background: #2563eb;
    }
    
    /* Remove default Streamlit styling */
    .stDeployButton {
        visibility: hidden;
    }
    
    header[data-testid="stHeader"] {
        height: 0;
    }
    
    .stMainBlockContainer {
        padding-top: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)


def create_header():
    """Create simple header."""
    st.markdown("""
    <div class="app-header">
        <h1 class="app-title">Financial Report Generator</h1>
        <p class="app-subtitle">AI-powered financial analysis and reporting</p>
    </div>
    """, unsafe_allow_html=True)


def create_sidebar():
    """Create sidebar with user inputs."""
    with st.sidebar:
        st.markdown("### Settings")
        
        # Check if we have API keys from environment
        try:
            from backend import ProjectManager
            pm = ProjectManager()
            has_openai, has_llama = pm.get_api_keys_status()
        except Exception as e:
            st.error(f"Error loading backend: {str(e)}")
            has_openai, has_llama = False, False
        
        # API Keys section
        st.markdown("#### API Keys")
        
        if has_openai and has_llama:
            st.success("✅ API keys loaded from environment")
            st.session_state.env_keys_available = True
        else:
            st.info("💡 Enter API keys or add them to .env file")
            st.session_state.env_keys_available = False
            
            openai_key = st.text_input(
                "OpenAI API Key",
                type="password",
                value=st.session_state.get('openai_key', ''),
                help="Required for AI analysis (starts with sk-)",
                placeholder="sk-..."
            )
            
            llama_key = st.text_input(
                "LlamaCloud API Key", 
                type="password",
                value=st.session_state.get('llama_key', ''),
                help="Required for document indexing (starts with llx-)",
                placeholder="llx-..."
            )
            
            # Validate API key formats
            if openai_key:
                if not openai_key.startswith('sk-'):
                    st.warning("⚠️ OpenAI API key should start with 'sk-'")
                st.session_state.openai_key = openai_key
            if llama_key:
                if not llama_key.startswith('llx-'):
                    st.warning("⚠️ LlamaCloud API key should start with 'llx-'")
                st.session_state.llama_key = llama_key
        
        st.markdown("---")
        
        # Project settings
        st.markdown("#### Project Settings")
        
        project_type = st.radio(
            "Project Type",
            ["Existing Project", "New Project"],
            help="Connect to existing or create new"
        )
        
        if project_type == "Existing Project":
            project_name = st.text_input(
                "Project Name",
                value="Default",
                help="Your LlamaCloud project name",
                disabled=True
            )
            
            index_name = st.text_input(
                "Index Name", 
                value="apple_tesla_demo_2",
                help="Index within the project"
            )
            
            # Validate project inputs
            if project_name and not project_name.strip():
                st.warning("⚠️ Project name cannot be empty")
            if index_name and not index_name.strip():
                st.warning("⚠️ Index name cannot be empty")
            
            st.session_state.project_config = {
                'type': 'existing',
                'project_name': project_name.strip() if project_name else '',
                'index_name': index_name.strip() if index_name else ''
            }
            
        else:
            new_project_name = st.text_input(
                "New Project Name",
                value="Default",
                help="Project name (using Default for LlamaCloud)",
                disabled=True
            )
            
            new_index_name = st.text_input(
                "New Index Name",
                value="financial_documents",
                help="Name for your document index (no spaces or special characters)"
            )
            
            # Validate new project inputs
            if new_project_name and not new_project_name.strip():
                st.warning("⚠️ Project name cannot be empty")
            if new_index_name and not new_index_name.strip():
                st.warning("⚠️ Index name cannot be empty")
            
            upload_method = st.selectbox(
                "Upload Method",
                ["Upload Files", "Provide URLs"]
            )
            
            files_data = None
            if upload_method == "Upload Files":
                uploaded_files = st.file_uploader(
                    "Upload Documents",
                    type=['pdf', 'txt', 'docx'],
                    accept_multiple_files=True,
                    help="Upload PDF, TXT, or DOCX files (max 200MB each)"
                )
                if uploaded_files:
                    # Validate file sizes
                    total_size = sum(file.size for file in uploaded_files)
                    if total_size > 500 * 1024 * 1024:  # 500MB total limit
                        st.error("⚠️ Total file size exceeds 500MB limit")
                    else:
                        files_data = uploaded_files
                        st.success(f"✅ {len(uploaded_files)} files selected ({total_size / (1024*1024):.1f}MB)")
                
            else:
                url_text = st.text_area(
                    "Document URLs",
                    height=100,
                    placeholder="https://example.com/doc1.pdf\nhttps://example.com/doc2.pdf",
                    help="Enter one URL per line (must be publicly accessible)"
                )
                if url_text.strip():
                    urls = [url.strip() for url in url_text.strip().split('\n') if url.strip()]
                    # Validate URLs
                    valid_urls = []
                    for url in urls:
                        if url.startswith(('http://', 'https://')):
                            valid_urls.append(url)
                        else:
                            st.warning(f"⚠️ Invalid URL: {url}")
                    
                    if valid_urls:
                        files_data = valid_urls
                        st.success(f"✅ {len(valid_urls)} valid URLs found")
            
            st.session_state.project_config = {
                'type': 'new',
                'project_name': new_project_name.strip() if new_project_name else '',
                'index_name': new_index_name.strip() if new_index_name else '',
                'files_data': files_data
            }


def create_status_dashboard():
    """Create simple status dashboard."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        status = "connected" if st.session_state.get('initialized', False) else "disconnected"
        icon = "✅" if status == "connected" else "❌"
        label = "Connected" if status == "connected" else "Disconnected"
        
        st.markdown(f"""
        <div class="status-card {status}">
            <div class="status-number">{icon}</div>
            <div class="status-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        count = st.session_state.get('reports_generated', 0)
        st.markdown(f"""
        <div class="status-card">
            <div class="status-number">{count}</div>
            <div class="status-label">Reports Generated</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        avg_time = st.session_state.get('avg_generation_time', 0)
        st.markdown(f"""
        <div class="status-card">
            <div class="status-number">{avg_time:.1f}s</div>
            <div class="status-label">Avg Time</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        project = st.session_state.get('current_project', 'None')
        st.markdown(f"""
        <div class="status-card">
            <div class="status-number">📁</div>
            <div class="status-label">{project}</div>
        </div>
        """, unsafe_allow_html=True)


def create_setup_interface():
    """Create simplified setup interface."""
    st.markdown("""
    <div class="card">
        <div class="card-title">🚀 Project Setup</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Check if we have all required inputs
    config = st.session_state.get('project_config', {})
    
    # Check API keys
    has_keys = False
    if st.session_state.get('env_keys_available'):
        has_keys = True
        openai_key = None
        llama_key = None
    else:
        openai_key = st.session_state.get('openai_key', '')
        llama_key = st.session_state.get('llama_key', '')
        has_keys = bool(openai_key and llama_key)
    
    if not has_keys:
        st.markdown("""
        <div class="alert alert-info">
            ℹ️ Please provide API keys in the sidebar to continue
        </div>
        """, unsafe_allow_html=True)
        return None
    
    if not config:
        st.markdown("""
        <div class="alert alert-info">
            ℹ️ Please configure your project in the sidebar
        </div>
        """, unsafe_allow_html=True)
        return None
    
    # Validate configuration
    setup_ready = True
    error_messages = []
    
    if config.get('type') == 'existing':
        if not config.get('project_name', '').strip():
            error_messages.append("Project name is required")
            setup_ready = False
        if not config.get('index_name', '').strip():
            error_messages.append("Index name is required")
            setup_ready = False
    elif config.get('type') == 'new':
        if not config.get('project_name', '').strip():
            error_messages.append("Project name is required")
            setup_ready = False
        if not config.get('index_name', '').strip():
            error_messages.append("Index name is required")
            setup_ready = False
        if not config.get('files_data'):
            error_messages.append("Please upload files or provide URLs")
            setup_ready = False
    
    # Display errors
    if error_messages:
        for msg in error_messages:
            st.markdown(f"""
            <div class="alert alert-error">
                ❌ {msg}
            </div>
            """, unsafe_allow_html=True)
    
    # Show setup summary
    if setup_ready:
        st.markdown("### Setup Summary")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Type:** {config.get('type', 'unknown').title()}")
            st.write(f"**Project:** {config.get('project_name', 'N/A')}")
        with col2:
            st.write(f"**Index:** {config.get('index_name', 'N/A')}")
            if config.get('type') == 'new' and config.get('files_data'):
                file_count = len(config['files_data'])
                st.write(f"**Files:** {file_count} items")
    
    # Check if setup is in progress
    setup_in_progress = st.session_state.get('setup_in_progress', False)
    
    if st.button("🚀 Setup Project", 
                disabled=not setup_ready or setup_in_progress, 
                use_container_width=True):
        try:
            # Mark setup as in progress
            st.session_state.setup_in_progress = True
            
            if config['type'] == 'existing':
                return ('existing', openai_key, llama_key, 
                       config['project_name'], config['index_name'])
            else:
                return ('new', openai_key, llama_key,
                       config['project_name'], config['index_name'], 
                       config.get('files_data'))
        except Exception as e:
            st.session_state.setup_in_progress = False
            st.error(f"Setup failed: {str(e)}")
            return None
    
    if setup_in_progress:
        st.info("⏳ Setup in progress... Please wait.")
        
    return None


def create_main_interface():
    """Create simplified main interface."""
    st.markdown("""
    <div class="card">
        <div class="card-title">💭 Financial Analysis</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Simple query interface
    query = st.text_area(
        "What would you like to analyze?",
        height=100,
        placeholder="e.g., Compare Apple and Tesla's revenue growth over the past 3 years...",
        help="Ask any question about your financial documents"
    )
    
    col1, col2 = st.columns([1, 4])
    with col2:
        generate_button = st.button(
            "Generate Report",
            disabled=not query.strip(),
            use_container_width=True
        )
    
    return query, generate_button


def show_loading(message="Processing..."):
    """Show simple loading indicator."""
    with st.spinner(message):
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.02)
            progress_bar.progress(i + 1)
        progress_bar.empty()


def display_success_message(message):
    """Display success message."""
    st.markdown(f"""
    <div class="alert alert-success">
        ✅ {message}
    </div>
    """, unsafe_allow_html=True)


def display_error_message(message):
    """Display error message."""
    st.markdown(f"""
    <div class="alert alert-error">
        ❌ {message}
    </div>
    """, unsafe_allow_html=True)


def display_info_message(message):
    """Display info message."""
    st.markdown(f"""
    <div class="alert alert-info">
        ℹ️ {message}
    </div>
    """, unsafe_allow_html=True)


def display_report(report_output):
    """Display generated report with simple formatting."""
    st.markdown("""
    <div class="card">
        <div class="card-title">📊 Financial Report</div>
    </div>
    """, unsafe_allow_html=True)
    
    for block in report_output.blocks:
        if hasattr(block, 'text'):  # TextBlock
            st.markdown(f"""
            <div class="report-section">
                <div class="report-text">{block.text}</div>
            </div>
            """, unsafe_allow_html=True)
            
        elif hasattr(block, 'caption'):  # TableBlock
            st.markdown(f"""
            <div class="report-section">
                <h4>{block.caption}</h4>
            </div>
            """, unsafe_allow_html=True)
            
            # Display table
            df = block.to_df()
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            # Simple visualization
            try:
                create_simple_chart(df, block.caption)
            except:
                pass


def create_simple_chart(df, title):
    """Create simple chart from dataframe."""
    try:
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        
        if len(numeric_cols) >= 1 and len(df) > 1:
            if len(df) <= 10:
                fig = px.bar(df, x=df.columns[0], y=numeric_cols[0], 
                           title=title, color_discrete_sequence=['#2563eb'])
            else:
                fig = px.line(df, x=df.columns[0], y=numeric_cols[0], 
                            title=title, color_discrete_sequence=['#2563eb'])
            
            fig.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white',
                font_color='#374151'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
    except Exception:
        pass


def initialize_session_state():
    """Initialize session state variables."""
    defaults = {
        'initialized': False,
        'generator': None,
        'reports_generated': 0,
        'avg_generation_time': 0.0,
        'generation_times': [],
        'current_project': 'None',
        'setup_complete': False,
        'env_keys_available': False,
        'setup_in_progress': False
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value