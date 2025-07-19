import os
import asyncio
import nest_asyncio
from typing import List, Tuple, Any
import pandas as pd
from llama_index.core.llms.llm import ToolSelection
from pydantic import BaseModel, Field

# Apply nest_asyncio for async support
nest_asyncio.apply()

# LlamaIndex imports
from llama_index.core import Settings, set_global_tokenizer
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


class FinancialReportGenerator:
    """Main class for financial report generation."""

    def __init__(self):
        self.index = None
        self.agent = None
        self.chunk_retriever = None
        self.doc_retriever = None
        self.is_initialized = False

    def initialize(self, openai_api_key: str, llama_cloud_api_key: str,
                   index_name: str = "apple_tesla_demo_2",
                   project_name: str = "llamacloud_demo"):
        """Initialize the report generator with API keys."""
        try:
            # Set environment variables
            os.environ["OPENAI_API_KEY"] = openai_api_key
            os.environ["LLAMA_CLOUD_API_KEY"] = llama_cloud_api_key

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
            return True, "Successfully initialized!"

        except Exception as e:
            return False, f"Initialization failed: {str(e)}"

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