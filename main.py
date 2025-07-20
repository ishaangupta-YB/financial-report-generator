import streamlit as st
import asyncio
import time
from backend import FinancialReportGenerator
from ui import (
    apply_custom_css, create_header, create_status_dashboard, create_setup_interface,
    create_main_interface, show_loading, display_report, display_success_message,
    display_error_message, display_info_message, create_sidebar,
    initialize_session_state
)

# Configure Streamlit page
st.set_page_config(
    page_title="Financial Report Generator",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)


def handle_project_setup(setup_type, openai_key, llama_key, project_name, index_name, files_data=None):
    """Handle project setup (existing or new)."""
    
    try:
        with st.container():
            if setup_type == "existing":
                show_loading("Connecting to existing project...")
                
                if 'generator' not in st.session_state or st.session_state.generator is None:
                    st.session_state.generator = FinancialReportGenerator()

                # Validate inputs
                if not project_name or not project_name.strip():
                    display_error_message("Project name cannot be empty")
                    return
                if not index_name or not index_name.strip():
                    display_error_message("Index name cannot be empty")
                    return

                success, message = st.session_state.generator.initialize_with_existing_index(
                    openai_key, llama_key, index_name.strip(), project_name.strip()
                )

                if success:
                    st.session_state.initialized = True
                    st.session_state.current_project = project_name.strip()
                    st.session_state.setup_complete = True
                    st.session_state.setup_in_progress = False
                    display_success_message(f"Successfully connected to project '{project_name}'!")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.session_state.setup_in_progress = False
                    display_error_message(f"Connection failed: {message}")

            elif setup_type == "new":
                show_loading("Creating new project...")
                
                if 'generator' not in st.session_state or st.session_state.generator is None:
                    st.session_state.generator = FinancialReportGenerator()

                # Validate inputs
                if not project_name or not project_name.strip():
                    display_error_message("Project name cannot be empty")
                    return
                if not index_name or not index_name.strip():
                    display_error_message("Index name cannot be empty")
                    return
                if not files_data:
                    display_error_message("No files provided for new project")
                    return

                # Handle different file types
                file_urls = None
                uploaded_files = None

                if isinstance(files_data, list) and all(isinstance(item, str) for item in files_data):
                    file_urls = files_data
                else:
                    uploaded_files = files_data

                success, message = st.session_state.generator.create_new_project(
                    openai_key, llama_key, index_name.strip(), project_name.strip(), file_urls, uploaded_files
                )

                if success:
                    st.session_state.initialized = True
                    st.session_state.current_project = project_name.strip()
                    st.session_state.setup_complete = True
                    st.session_state.setup_in_progress = False
                    display_success_message(f"Successfully created project '{project_name}'!")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.session_state.setup_in_progress = False
                    display_error_message(f"Project creation failed: {message}")
    
    except Exception as e:
        st.session_state.setup_in_progress = False
        display_error_message(f"Setup error: {str(e)}")
        # Reset generator on error
        if 'generator' in st.session_state:
            st.session_state.generator = None


def handle_report_generation(query):
    """Handle report generation process."""
    # Validate inputs
    if not query or not query.strip():
        display_error_message("Query cannot be empty")
        return
    
    if not st.session_state.get('generator'):
        display_error_message("Generator not initialized. Please set up your project first.")
        return
    
    if not st.session_state.get('initialized'):
        display_error_message("Project not initialized. Please complete setup first.")
        return

    start_time = time.time()
    query = query.strip()

    show_loading("Generating financial report...")

    try:
        # Run async report generation with timeout
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Add timeout to prevent hanging
        try:
            success, message, report = loop.run_until_complete(
                asyncio.wait_for(
                    st.session_state.generator.generate_report(query),
                    timeout=300  # 5 minute timeout
                )
            )
        except asyncio.TimeoutError:
            success, message, report = False, "Report generation timed out (5 minutes)", None
        finally:
            loop.close()

        if success and report:
            # Update metrics
            generation_time = time.time() - start_time
            st.session_state.reports_generated += 1
            if 'generation_times' not in st.session_state:
                st.session_state.generation_times = []
            st.session_state.generation_times.append(generation_time)
            st.session_state.avg_generation_time = (
                    sum(st.session_state.generation_times) / len(st.session_state.generation_times)
            )

            display_success_message("Your financial analysis report is ready!")
            display_report(report)

            # Add download functionality
            st.markdown("---")
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                try:
                    report_text = generate_text_report(report)
                    st.download_button(
                        label="📥 Download Report",
                        data=report_text,
                        file_name=f"financial_report_{int(time.time())}.txt",
                        mime="text/plain",
                        help="Download your report as a text file",
                        use_container_width=True
                    )
                except Exception as e:
                    st.warning(f"Download generation failed: {str(e)}")
        else:
            display_error_message(f"Report generation failed: {message or 'Unknown error'}")

    except Exception as e:
        display_error_message(f"Unexpected error: {str(e)}")
        # Log the error for debugging
        st.error(f"Debug info: {type(e).__name__}: {str(e)}")


def generate_text_report(report_output):
    """Generate downloadable text version of report."""
    try:
        if not report_output or not hasattr(report_output, 'blocks'):
            return "Error: Invalid report format"
        
        lines = [
            "=" * 80,
            "FINANCIAL ANALYSIS REPORT",
            "=" * 80,
            f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Project: {st.session_state.get('current_project', 'Unknown')}",
            "=" * 80,
            ""
        ]

        if not report_output.blocks:
            lines.extend([
                "No content available in this report.",
                ""
            ])
        else:
            for i, block in enumerate(report_output.blocks, 1):
                try:
                    if hasattr(block, 'text'):  # TextBlock
                        lines.extend([
                            f"SECTION {i}: TEXT ANALYSIS",
                            "-" * 40,
                            str(block.text) if block.text else "No text content",
                            ""
                        ])
                    elif hasattr(block, 'caption'):  # TableBlock
                        caption = str(block.caption) if block.caption else f"Table {i}"
                        lines.extend([
                            f"SECTION {i}: {caption.upper()}",
                            "-" * 40,
                            ""
                        ])

                        # Add table with error handling
                        if hasattr(block, 'col_names') and hasattr(block, 'rows'):
                            if block.col_names and block.rows:
                                header = " | ".join(f"{str(col):>15}" for col in block.col_names)
                                lines.append(header)
                                lines.append("-" * len(header))

                                for row in block.rows:
                                    if row:  # Check if row is not None/empty
                                        row_text = " | ".join(f"{str(cell):>15}" for cell in row)
                                        lines.append(row_text)
                            else:
                                lines.append("No table data available")
                        else:
                            lines.append("Invalid table format")
                        
                        lines.append("")
                        
                except Exception as e:
                    lines.extend([
                        f"SECTION {i}: ERROR",
                        "-" * 40,
                        f"Error processing block: {str(e)}",
                        ""
                    ])

        lines.extend([
            "=" * 80,
            "END OF REPORT",
            "=" * 80
        ])

        return "\n".join(lines)
    
    except Exception as e:
        return f"Error generating report text: {str(e)}"


def main():
    """Main application function."""
    try:
        # Initialize session state
        initialize_session_state()

        # Apply custom styling
        apply_custom_css()

        # Create sidebar
        create_sidebar()

        # Create header
        create_header()

        # Create status dashboard
        create_status_dashboard()

        # Main application logic
        if not st.session_state.get('setup_complete', False):
            # Show setup interface
            setup_result = create_setup_interface()

            if setup_result:  # Valid setup action
                try:
                    setup_type = setup_result[0]
                    openai_key = setup_result[1]
                    llama_key = setup_result[2]
                    project_name = setup_result[3]
                    index_name = setup_result[4]
                    files_data = setup_result[5] if len(setup_result) > 5 else None

                    handle_project_setup(setup_type, openai_key, llama_key, project_name, index_name, files_data)
                except (IndexError, TypeError) as e:
                    display_error_message(f"Invalid setup configuration: {str(e)}")

        else:
            # Main interface for report generation
            if st.session_state.get('initialized', False):
                try:
                    query, generate_button = create_main_interface()

                    if generate_button:
                        if query and query.strip():
                            handle_report_generation(query)
                        else:
                            display_error_message("Please enter a query before generating a report.")
                except Exception as e:
                    display_error_message(f"Interface error: {str(e)}")

                # Add reset option
                st.markdown("---")
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    if st.button("🔄 Reset Project", help="Start over with a new project setup", use_container_width=True):
                        try:
                            # Reset session state
                            keys_to_reset = ['initialized', 'setup_complete', 'current_project', 'generator', 'project_config', 
                                           'openai_key', 'llama_key', 'env_keys_available', 'setup_in_progress']
                            for key in keys_to_reset:
                                if key in st.session_state:
                                    del st.session_state[key]
                            display_info_message("Project reset successfully!")
                            time.sleep(1)
                            st.rerun()
                        except Exception as e:
                            display_error_message(f"Reset failed: {str(e)}")

            else:
                display_error_message("Setup completed but initialization failed. Please try resetting the project.")
                
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.error("Please refresh the page to restart the application.")


if __name__ == "__main__":
    main()