import streamlit as st
import asyncio
import time
from backend import FinancialReportGenerator
from ui import (
    apply_custom_css, create_header, create_sidebar, create_status_indicators,
    create_query_interface, display_loading_animation, display_report,
    display_success_message, display_error_message, create_footer,
    initialize_session_state
)

# Configure Streamlit page
st.set_page_config(
    page_title="Financial Report Generator",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)


def main():
    """Main application function."""
    # Initialize session state
    initialize_session_state()

    # Apply custom styling
    apply_custom_css()

    # Create header
    create_header()

    # Create sidebar and get configuration
    openai_key, llama_cloud_key, index_name, project_name, selected_example = create_sidebar()

    # Initialize generator if keys are provided
    if openai_key and llama_cloud_key and not st.session_state.initialized:
        with st.spinner("🔧 Initializing AI models and connections..."):
            if 'generator' not in st.session_state or st.session_state.generator is None:
                st.session_state.generator = FinancialReportGenerator()

            success, message = st.session_state.generator.initialize(
                openai_key, llama_cloud_key, index_name, project_name
            )

            if success:
                st.session_state.initialized = True
                st.success(message)
            else:
                st.error(message)
                st.session_state.initialized = False

    # Create status indicators
    create_status_indicators()

    # Create main interface
    if st.session_state.initialized:
        # Query interface
        query, generate_button = create_query_interface()

        # Handle report generation
        if generate_button and query.strip():
            start_time = time.time()

            # Show loading animation
            loading_placeholder = st.empty()
            with loading_placeholder.container():
                display_loading_animation()

            # Generate report
            try:
                # Run async function
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                success, message, report = loop.run_until_complete(
                    st.session_state.generator.generate_report(query)
                )
                loop.close()

                # Clear loading animation
                loading_placeholder.empty()

                if success:
                    # Update metrics
                    generation_time = time.time() - start_time
                    st.session_state.reports_generated += 1
                    st.session_state.generation_times.append(generation_time)
                    st.session_state.avg_generation_time = sum(st.session_state.generation_times) / len(
                        st.session_state.generation_times)

                    # Display success message
                    display_success_message()

                    # Display report
                    display_report(report)

                    # Add download button
                    st.markdown("---")
                    col1, col2, col3 = st.columns([2, 1, 2])
                    with col2:
                        if st.button("📥 Download Report", help="Download report as text file"):
                            report_text = generate_text_report(report)
                            st.download_button(
                                label="💾 Save as TXT",
                                data=report_text,
                                file_name=f"financial_report_{int(time.time())}.txt",
                                mime="text/plain"
                            )

                else:
                    display_error_message(message)

            except Exception as e:
                loading_placeholder.empty()
                display_error_message(f"Unexpected error: {str(e)}")

        elif generate_button and not query.strip():
            st.warning("⚠️ Please enter a query before generating a report.")

    else:
        # Show setup instructions
        st.markdown("""
        <div class="feature-card">
            <h3>🚀 Getting Started</h3>
            <p>To begin generating financial reports, please provide your API keys in the sidebar:</p>
            <ul>
                <li><strong>OpenAI API Key:</strong> Required for AI-powered analysis</li>
                <li><strong>LlamaCloud API Key:</strong> Required for document processing</li>
            </ul>
            <p>Once connected, you'll be able to generate comprehensive financial reports with just a few clicks!</p>
        </div>
        """, unsafe_allow_html=True)

        # Show features
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            <div class="feature-card">
                <h4>🔍 Smart Analysis</h4>
                <p>AI-powered financial document analysis with intelligent information retrieval.</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="feature-card">
                <h4>📊 Rich Reports</h4>
                <p>Generate comprehensive reports with text, tables, and visualizations.</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="feature-card">
                <h4>⚡ Fast Processing</h4>
                <p>Quick retrieval and analysis of complex financial documents.</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="feature-card">
                <h4>🔒 Secure</h4>
                <p>Your API keys and data are handled securely and never stored.</p>
            </div>
            """, unsafe_allow_html=True)

    # Create footer
    create_footer()


def generate_text_report(report_output):
    """Generate a text version of the report for download."""
    text_content = []
    text_content.append("FINANCIAL ANALYSIS REPORT")
    text_content.append("=" * 50)
    text_content.append("")

    for i, block in enumerate(report_output.blocks):
        if hasattr(block, 'text'):  # TextBlock
            text_content.append(block.text)
            text_content.append("")
        elif hasattr(block, 'caption'):  # TableBlock
            text_content.append(f"TABLE: {block.caption}")
            text_content.append("-" * len(f"TABLE: {block.caption}"))

            # Add table headers
            text_content.append(" | ".join(block.col_names))
            text_content.append("-" * (len(" | ".join(block.col_names))))

            # Add table rows
            for row in block.rows:
                text_content.append(" | ".join(str(cell) for cell in row))
            text_content.append("")

    text_content.append("-" * 50)
    text_content.append(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    return "\n".join(text_content)


if __name__ == "__main__":
    main()