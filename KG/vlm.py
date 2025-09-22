from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline
from docling.datamodel.pipeline_options import VlmPipelineOptions
from docling.datamodel import vlm_model_specs

# Configure VLM pipeline
pipeline_options = VlmPipelineOptions(
    vlm_options=vlm_model_specs.SMOLDOCLING_MLX  # Use SmolDocling model
)

# Initialize converter with VLM pipeline
converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_cls=VlmPipeline,
            pipeline_options=pipeline_options
        )
    }
)

# Convert document using VLM
doc = converter.convert("policy.pdf").document
