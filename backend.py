import os
import asyncio
import nest_asyncio
import requests
import tempfile
import wget
import uuid
from typing import List, Tuple, Any, Optional
import pandas as pd
from llama_index.core.llms.llm import ToolSelection
from pydantic import BaseModel, Field
import time
from dotenv import load_dotenv
from urllib.parse import urlparse
import mimetypes

# Load environment variables from .env file
load_dotenv()

# Apply nest_asyncio for async support
nest_asyncio.apply()

# LlamaIndex imports
from llama_index.core import Settings, set_global_tokenizer, SimpleDirectoryReader, Document
from llama_index.core.workflow import Workflow, StartEvent, StopEvent, Context, step, Event
from llama_index.core.tools import FunctionTool, BaseTool
from llama_index.core.schema import NodeWithScore
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.llms import ChatMessage
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.indices.managed.llama_cloud import LlamaCloudIndex
from llama_index.core.response_synthesizers import TreeSummarize, CompactAndRefine
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.llms.structured_llm import StructuredLLM

# Set tokenizer
import tiktoken

set_global_tokenizer(tiktoken.encoding_for_model("gpt-4o").encode)


class TextBlock(BaseModel):
    """Text block for reports."""
    text: str = Field(..., description="The text for this block.")


class TableBlock(BaseModel):
    """Table block for reports."""
    caption: str = Field(..., description="Caption of the table.")
    col_names: List[str] = Field(..., description="Names of the columns.")
    rows: List[Tuple] = Field(..., description="List of rows as tuples.")

    def to_df(self) -> pd.DataFrame:
        """Convert to pandas DataFrame."""
        df = pd.DataFrame(self.rows, columns=self.col_names)
        return df


class ReportOutput(BaseModel):
    """Data model for a complete report."""
    blocks: List[TextBlock | TableBlock] = Field(..., description="List of text and table blocks.")


# Event classes for workflow
class InputEvent(Event):
    input: List[ChatMessage]


class ChunkRetrievalEvent(Event):
    tool_call: ToolSelection


class DocRetrievalEvent(Event):
    tool_call: ToolSelection


class ReportGenerationEvent(Event):
    pass


class ProjectManager:
    """Manages LlamaCloud projects and indexes."""

    def __init__(self):
        # Try to get API keys from environment variables first
        self.llama_cloud_api_key = os.getenv("LLAMA_CLOUD_API_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")

    def set_api_keys(self, openai_key: str = None, llama_cloud_key: str = None):
        """Set API keys for services. Uses environment variables if keys not provided."""
        # Use provided keys or fall back to environment variables
        self.openai_api_key = openai_key or self.openai_api_key
        self.llama_cloud_api_key = llama_cloud_key or self.llama_cloud_api_key
        
        if self.openai_api_key:
            os.environ["OPENAI_API_KEY"] = self.openai_api_key
        if self.llama_cloud_api_key:
            os.environ["LLAMA_CLOUD_API_KEY"] = self.llama_cloud_api_key

    def get_api_keys_status(self) -> tuple[bool, bool]:
        """Check if API keys are available."""
        return bool(self.openai_api_key), bool(self.llama_cloud_api_key)

    def validate_index_exists(self, index_name: str, project_name: str) -> tuple[bool, str]:
        """Validate if index exists in LlamaCloud."""
        try:
            index = LlamaCloudIndex(
                name=index_name,
                project_name=project_name,
                api_key=self.llama_cloud_api_key
            )
            # Try to get retriever to test if index exists
            retriever = index.as_retriever(retrieval_mode="chunks", rerank_top_n=1)
            return True, f"Index '{index_name}' found in project '{project_name}'"
        except Exception as e:
            return False, f"Index validation failed: {str(e)}"

    def download_file_from_url(self, url: str, filename: str = None) -> tuple[bool, str, str]:
        """Download file from URL to temporary directory using multiple methods."""
        try:
            # Parse URL to get a better filename if not provided
            parsed_url = urlparse(url)
            if not filename:
                # Extract filename from URL or create one
                url_filename = os.path.basename(parsed_url.path)
                if url_filename and '.' in url_filename:
                    filename = url_filename
                else:
                    # Generate filename based on content type
                    try:
                        head_response = requests.head(url, timeout=10)
                        content_type = head_response.headers.get('content-type', '').lower()
                        if 'pdf' in content_type:
                            filename = f"document_{uuid.uuid4().hex[:8]}.pdf"
                        elif 'doc' in content_type:
                            filename = f"document_{uuid.uuid4().hex[:8]}.docx"
                        elif 'text' in content_type:
                            filename = f"document_{uuid.uuid4().hex[:8]}.txt"
                        else:
                            filename = f"document_{uuid.uuid4().hex[:8]}.pdf"
                    except:
                        filename = f"document_{uuid.uuid4().hex[:8]}.pdf"

            # Create temporary directory
            temp_dir = tempfile.mkdtemp()
            file_path = os.path.join(temp_dir, filename)

            # Try using wget first (more reliable for large files)
            try:
                wget.download(url, file_path)
                if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                    return True, f"Downloaded {filename} successfully ({os.path.getsize(file_path)} bytes)", file_path
            except Exception as wget_error:
                print(f"wget failed: {wget_error}, trying requests...")

            # Fallback to requests with streaming
            try:
                response = requests.get(url, timeout=60, stream=True)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                
                with open(file_path, 'wb') as f:
                    downloaded = 0
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                
                if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                    return True, f"Downloaded {filename} successfully ({os.path.getsize(file_path)} bytes)", file_path
                else:
                    return False, f"Downloaded file is empty or corrupted", ""
                    
            except Exception as requests_error:
                return False, f"Both download methods failed. wget: {str(wget_error) if 'wget_error' in locals() else 'N/A'}, requests: {str(requests_error)}", ""

        except Exception as e:
            return False, f"Failed to download from {url}: {str(e)}", ""

    def create_documents_from_files(self, file_paths: List[str]) -> tuple[bool, str, List[Document]]:
        """Create LlamaIndex documents from file paths."""
        try:
            documents = []
            for file_path in file_paths:
                if os.path.exists(file_path):
                    reader = SimpleDirectoryReader(input_files=[file_path])
                    docs = reader.load_data()
                    documents.extend(docs)

            if documents:
                return True, f"Created {len(documents)} documents from {len(file_paths)} files", documents
            else:
                return False, "No documents were created from the provided files", []
        except Exception as e:
            return False, f"Failed to create documents: {str(e)}", []

    def create_new_index(self, index_name: str, project_name: str, documents: List[Document]) -> tuple[bool, str]:
        """Create a new index in LlamaCloud with documents using the proper API."""
        try:
            # Always use "Default" as the project name as instructed
            project_name = "Default"
            
            # Import here to avoid circular imports
            from llama_index.indices.managed.llama_cloud import LlamaCloudIndex
            
            # Create index using the from_documents method
            index = LlamaCloudIndex.from_documents(
                documents=documents,
                name=index_name,
                project_name=project_name,
                api_key=self.llama_cloud_api_key,
                verbose=True
            )
            
            # Verify the index was created
            if index:
                return True, f"Successfully created index '{index_name}' in project '{project_name}' with {len(documents)} documents"
            else:
                return False, "Index creation returned None"
                
        except Exception as e:
            error_msg = str(e)
            if "Project" in error_msg and "not found" in error_msg:
                return False, f"Project '{project_name}' not found. Please create the project first in LlamaCloud dashboard or use 'Default' project."
            return False, f"Failed to create index: {error_msg}"


class FinancialReportGenerator:
    """Main class for financial report generation."""

    def __init__(self):
        self.index = None
        self.agent = None
        self.chunk_retriever = None
        self.doc_retriever = None
        self.is_initialized = False
        self.project_manager = ProjectManager()

    def initialize_with_existing_index(self, openai_api_key: str, llama_cloud_api_key: str,
                                       index_name: str, project_name: str) -> tuple[bool, str]:
        """Initialize with existing index."""
        try:
            self.project_manager.set_api_keys(openai_api_key, llama_cloud_api_key)

            # Validate index exists
            exists, message = self.project_manager.validate_index_exists(index_name, project_name)
            if not exists:
                return False, message

            # Setup models
            embed_model = OpenAIEmbedding(model="text-embedding-3-large")
            llm = OpenAI(model="gpt-4o-mini")

            Settings.embed_model = embed_model
            Settings.llm = llm

            # Setup LlamaCloud index
            self.index = LlamaCloudIndex(
                name=index_name,
                project_name=project_name,
                api_key=llama_cloud_api_key
            )

            # Setup retrievers
            self.doc_retriever = self.index.as_retriever(
                retrieval_mode="files_via_content",
                files_top_k=1
            )

            self.chunk_retriever = self.index.as_retriever(
                retrieval_mode="chunks",
                rerank_top_n=5
            )

            # Create tools
            chunk_retriever_tool = FunctionTool.from_defaults(fn=self._chunk_retriever_fn)
            doc_retriever_tool = FunctionTool.from_defaults(fn=self._doc_retriever_fn)

            # Setup report generation LLM
            report_gen_system_prompt = """
You are a report generation assistant tasked with producing well-formatted financial reports.
You will be given context from financial documents and must produce a report with interleaving text and tables.

Make sure the report is detailed with textual explanations, especially when tables are provided.
Use table blocks to present quantitative metrics and comparisons.

You MUST output your response as structured data matching the ReportOutput format.
"""

            report_gen_llm = OpenAI(model="gpt-4o", max_tokens=2048, system_prompt=report_gen_system_prompt)
            report_gen_sllm = report_gen_llm.as_structured_llm(output_cls=ReportOutput)

            # Create agent
            self.agent = ReportGenerationAgent(
                chunk_retriever_tool,
                doc_retriever_tool,
                llm=llm,
                report_gen_sllm=report_gen_sllm,
                verbose=False,
                timeout=120.0,
            )

            self.is_initialized = True
            return True, f"Successfully connected to existing index '{index_name}'"

        except Exception as e:
            return False, f"Initialization failed: {str(e)}"

    def create_new_project(self, openai_api_key: str, llama_cloud_api_key: str,
                           index_name: str, project_name: str,
                           file_urls: List[str] = None, uploaded_files=None) -> tuple[bool, str]:
        """Create new project with files."""
        try:
            self.project_manager.set_api_keys(openai_api_key, llama_cloud_api_key)

            # Force project name to "Default"
            project_name = "Default"
            
            file_paths = []
            download_messages = []

            # Handle URL downloads
            if file_urls:
                for i, url in enumerate(file_urls):
                    print(f"Downloading file {i+1}/{len(file_urls)}: {url}")
                    success, message, file_path = self.project_manager.download_file_from_url(url)
                    download_messages.append(message)
                    if success:
                        file_paths.append(file_path)
                        print(f"✓ {message}")
                    else:
                        print(f"✗ {message}")
                        return False, f"Download failed: {message}"

            # Handle uploaded files
            if uploaded_files:
                temp_dir = tempfile.mkdtemp()
                for uploaded_file in uploaded_files:
                    try:
                        file_path = os.path.join(temp_dir, uploaded_file.name)
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        # Validate file was written correctly
                        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                            file_paths.append(file_path)
                            print(f"✓ Saved uploaded file: {uploaded_file.name}")
                        else:
                            return False, f"Failed to save uploaded file: {uploaded_file.name}"
                    except Exception as e:
                        return False, f"Error processing uploaded file {uploaded_file.name}: {str(e)}"

            if not file_paths:
                return False, "No files provided for index creation"

            print(f"Processing {len(file_paths)} files for document creation...")

            # Create documents
            success, message, documents = self.project_manager.create_documents_from_files(file_paths)
            if not success:
                return False, f"Document creation failed: {message}"

            print(f"Created {len(documents)} documents, creating LlamaCloud index...")

            # Create index with proper error handling
            success, message = self.project_manager.create_new_index(index_name, project_name, documents)
            if not success:
                return False, f"Index creation failed: {message}"

            print(f"Index created successfully, initializing connection...")

            # Now initialize with the new index
            init_success, init_message = self.initialize_with_existing_index(openai_api_key, llama_cloud_api_key, index_name, project_name)
            if init_success:
                return True, f"Project created successfully with {len(documents)} documents"
            else:
                return False, f"Index created but initialization failed: {init_message}"

        except Exception as e:
            return False, f"Project creation failed: {str(e)}"

    def _chunk_retriever_fn(self, query: str) -> List[NodeWithScore]:
        """Retrieves relevant document chunks."""
        return self.chunk_retriever.retrieve(query)

    def _doc_retriever_fn(self, query: str) -> List[NodeWithScore]:
        """Retrieves entire documents."""
        return self.doc_retriever.retrieve(query)

    async def generate_report(self, query: str) -> tuple[bool, str, ReportOutput]:
        """Generate a financial report based on the query."""
        if not self.is_initialized:
            return False, "Generator not initialized", None

        try:
            result = await self.agent.run(input=query)
            report = result["response"].response
            return True, "Report generated successfully", report
        except Exception as e:
            return False, f"Report generation failed: {str(e)}", None


class ReportGenerationAgent(Workflow):
    """Agent workflow for report generation."""

    def __init__(
            self,
            chunk_retriever_tool: BaseTool,
            doc_retriever_tool: BaseTool,
            llm: FunctionCallingLLM = None,
            report_gen_sllm: StructuredLLM = None,
            **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.chunk_retriever_tool = chunk_retriever_tool
        self.doc_retriever_tool = doc_retriever_tool
        self.llm = llm or OpenAI()
        self.summarizer = CompactAndRefine(llm=self.llm)
        self.report_gen_sllm = report_gen_sllm
        self.report_gen_summarizer = TreeSummarize(llm=self.report_gen_sllm)
        self.memory = ChatMemoryBuffer.from_defaults(llm=self.llm)
        self.sources = []

    @step(pass_context=True)
    async def prepare_chat_history(self, ctx: Context, ev: StartEvent) -> InputEvent:
        self.sources = []
        ctx.data["stored_chunks"] = []
        ctx.data["query"] = ev.input

        user_msg = ChatMessage(role="user", content=ev.input)
        self.memory.put(user_msg)

        chat_history = self.memory.get()
        return InputEvent(input=chat_history)

    @step(pass_context=True)
    async def handle_llm_input(
            self, ctx: Context, ev: InputEvent
    ) -> ChunkRetrievalEvent | DocRetrievalEvent | ReportGenerationEvent | StopEvent:
        chat_history = ev.input

        response = await self.llm.achat_with_tools(
            [self.chunk_retriever_tool, self.doc_retriever_tool],
            chat_history=chat_history,
        )
        self.memory.put(response.message)

        tool_calls = self.llm.get_tool_calls_from_response(
            response, error_on_no_tool_call=False
        )

        if not tool_calls:
            return ReportGenerationEvent()

        for tool_call in tool_calls:
            if tool_call.tool_name == self.chunk_retriever_tool.metadata.name:
                return ChunkRetrievalEvent(tool_call=tool_call)
            elif tool_call.tool_name == self.doc_retriever_tool.metadata.name:
                return DocRetrievalEvent(tool_call=tool_call)
            else:
                return StopEvent(result={"response": "Invalid tool."})

    @step(pass_context=True)
    async def handle_retrieval(
            self, ctx: Context, ev: ChunkRetrievalEvent | DocRetrievalEvent
    ) -> InputEvent:
        query = ev.tool_call.tool_kwargs["query"]
        if isinstance(ev, ChunkRetrievalEvent):
            retrieved_chunks = self.chunk_retriever_tool(query).raw_output
        else:
            retrieved_chunks = self.doc_retriever_tool(query).raw_output
        ctx.data["stored_chunks"].extend(retrieved_chunks)

        response = self.summarizer.synthesize(query, nodes=retrieved_chunks)
        self.memory.put(
            ChatMessage(
                role="tool",
                content=str(response),
                additional_kwargs={
                    "tool_call_id": ev.tool_call.tool_id,
                    "name": ev.tool_call.tool_name,
                },
            )
        )

        return InputEvent(input=self.memory.get())

    @step(pass_context=True)
    async def generate_report(
            self, ctx: Context, ev: ReportGenerationEvent
    ) -> StopEvent:
        response = self.report_gen_summarizer.synthesize(
            ctx.data["query"], nodes=ctx.data["stored_chunks"]
        )
        return StopEvent(result={"response": response})