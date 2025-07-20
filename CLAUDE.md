# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Financial Report Generator** - a Streamlit-based AI application that generates intelligent financial analysis reports from documents using LlamaIndex and OpenAI. The application can process financial documents (PDFs, TXT, DOCX) and generate comprehensive reports with tables, visualizations, and textual analysis.

## Architecture

### Core Components

1. **main.py** - Streamlit application entry point and orchestration
   - Handles session state management with comprehensive error handling
   - Coordinates between UI and backend components
   - Manages the report generation workflow with timeouts and validation
   - Provides async support for report generation with error recovery

2. **backend.py** - Core AI/ML functionality with environment variable support
   - `FinancialReportGenerator` - Main class for report generation
   - `ProjectManager` - Handles LlamaCloud projects, document indexing, and API key management
   - `ReportGenerationAgent` - Workflow-based agent for AI-powered analysis
   - Data models: `TextBlock`, `TableBlock`, `ReportOutput` for structured output
   - Environment variable support with `.env` file integration

3. **ui.py** - Minimal, accessible UI components and styling
   - Clean, high-contrast CSS design (white background, dark text)
   - Sidebar-based configuration interface
   - Comprehensive input validation and error messaging
   - Simple status dashboard and loading indicators
   - Report display with enhanced error handling

### Data Flow

1. **Environment Setup**: Application checks for API keys in `.env` file first, falls back to user input
2. **Sidebar Configuration**: User configures project settings in sidebar (existing vs new project)
3. **Validation**: Comprehensive input validation with helpful error messages
4. **Setup Phase**: User connects to existing project or creates new one with file validation
5. **Indexing**: Documents are processed and indexed in LlamaCloud for retrieval
6. **Query Processing**: User query triggers retrieval of relevant document chunks/sections
7. **AI Analysis**: OpenAI models analyze retrieved content and generate structured reports
8. **Display**: Reports rendered with tables, text blocks, and simple visualizations

### Key Dependencies

- **Streamlit** - Web application framework
- **LlamaIndex** - Document indexing, retrieval, and AI orchestration
- **OpenAI** - LLM for analysis and report generation
- **Plotly** - Interactive visualizations
- **Pandas** - Data manipulation for tables
- **Pydantic** - Data validation and structured output
- **python-dotenv** - Environment variable management

## Development Commands

### Running the Application
```bash
streamlit run main.py
```

### Installing Dependencies
```bash
pip install -r requirements.txt
```

### Environment Setup
The application supports two methods for API key configuration:

1. **Environment Variables (Recommended)**:
   - Copy `.env.example` to `.env`
   - Add your API keys:
     ```
     OPENAI_API_KEY=sk-your-key-here
     LLAMA_CLOUD_API_KEY=llx-your-key-here
     ```

2. **Manual Input**: Enter keys through the sidebar interface

## Key Technical Details

### Enhanced Session State Management
The application uses Streamlit's session state to track:
- `initialized`: Whether the generator is ready
- `generator`: FinancialReportGenerator instance
- `setup_complete`: Setup workflow completion
- `reports_generated`: Metrics tracking
- `current_project`: Active project name
- `env_keys_available`: Environment variable status
- `project_config`: Sidebar configuration state

### Error Handling and Edge Cases
- **API Key Validation**: Format validation (sk-*, llx-*) with helpful warnings
- **Input Sanitization**: Trimming whitespace, validating required fields
- **File Upload Limits**: 500MB total limit with size validation
- **URL Validation**: Proper HTTP/HTTPS format checking
- **Timeout Handling**: 5-minute timeout for report generation
- **Connection Recovery**: Automatic generator reset on errors
- **Graceful Degradation**: Error messages with context and recovery suggestions

### Async Workflow
Report generation uses async/await patterns with `nest_asyncio` for Streamlit compatibility:
1. Prepare chat history
2. Handle LLM input with tool calling
3. Retrieve relevant chunks/documents
4. Generate structured report output
5. Timeout protection and error recovery

### Structured Output
Reports use Pydantic models for type safety:
- `ReportOutput` contains list of `TextBlock` and `TableBlock` objects
- Tables automatically convert to pandas DataFrames
- Visualizations are generated automatically from numeric table data
- Comprehensive error handling for malformed data

### Document Processing
- Supports PDF, TXT, DOCX file uploads with size validation
- Can download documents from URLs with accessibility checks
- Uses LlamaCloud for enterprise-grade document indexing
- Supports both chunk-level and document-level retrieval

## File Organization

```
├── main.py              # Application entry point and orchestration
├── backend.py           # AI/ML core functionality with env support
├── ui.py               # Clean, minimal UI components
├── requirements.txt    # Python dependencies
├── .env.example        # Environment variable template
└── CLAUDE.md          # This documentation file
```

## Common Patterns

### Error Handling
- Functions return `(success: bool, message: str, data: optional)` tuples
- Comprehensive try-catch blocks with user-friendly error messages
- Graceful degradation with recovery suggestions
- Debug information for development

### UI Components
- **Sidebar-driven configuration**: All settings centralized
- **Input validation**: Real-time validation with helpful warnings
- **Status feedback**: Clear success/error/info messages
- **Progressive disclosure**: Show configuration summary before setup

### API Integration
- **Environment-first approach**: Check `.env` before prompting user
- **API key validation**: Format checking with helpful feedback
- **Connection testing**: Validate credentials before proceeding
- **Error recovery**: Reset state and provide clear next steps

### Accessibility and UX
- **High contrast design**: White background with dark text
- **Clear visual hierarchy**: Simple cards and consistent spacing
- **Helpful placeholders**: Example inputs and format hints
- **Loading feedback**: Simple spinners with progress indication
- **Mobile-friendly**: Responsive design that works on different screen sizes

## Development Best Practices

### Adding New Features
1. **Validation First**: Add input validation and error handling
2. **Sidebar Integration**: Place configuration options in sidebar
3. **Error Messages**: Provide helpful, actionable error messages
4. **State Management**: Use session state consistently
5. **Testing**: Test edge cases like empty inputs, large files, network failures

### UI Guidelines
- Use the established color scheme (white/dark text/blue accents)
- Follow the card-based layout pattern
- Add validation with immediate feedback
- Include helpful placeholder text and tooltips
- Test with different screen sizes

### Error Handling Guidelines
- Always catch exceptions at the function level
- Provide user-friendly error messages
- Include context about what went wrong
- Suggest specific next steps for recovery
- Log technical details for debugging when needed