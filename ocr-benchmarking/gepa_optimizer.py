"""
GEPA Optimizer for OCR Benchmarking

This script uses DSPy's GEPA (Generalized Error-Prompt Alignment) optimizer to automatically
improve a two-stage OCR pipeline:
  1. Image → Markdown (vision model extracts text from document images)
  2. Markdown → JSON (language model extracts structured data based on a schema)

The optimizer refines the prompts and instructions for both stages by analyzing
failures and generating improved versions.
"""

import base64
import hashlib
import json
import os
import random
from pathlib import Path
from urllib.parse import urlparse

import dspy
from agno.models.openrouter import OpenRouter
from dspy import GEPA
from dspy.adapters.base import Adapter
from agno.agent import Agent
from dspy.signatures.signature import Signature

from open_router_client import OpenRouterClient
from parsing_utils import (
    generate_pydantic_model,
    run_custom_structured_output_agent_with_chain_of_thought,
)
from download_data import list_all_document_pairs
from scorer import call_typescript_json_evaluation, deterministic_json_metric

DEFAULT_TENACITY_RETRY_ATTEMPTS = 3

# Model configuration - can be changed to test different models
REFLECTION_MODEL = "openai/gpt-5"  # Used by GEPA for analyzing failures and proposing improvements
OCR_MODEL = "google/gemini-2.0-flash-001"  # Vision model for image→markdown conversion
JSON_MODEL = "openai/gpt-4.1"  # Language model for markdown→JSON extraction

REFLECTION_MODEL_ID = REFLECTION_MODEL
OCR_MODEL_ID = OCR_MODEL
OR_JSON_MODEL = JSON_MODEL

# Primary language model for the OCR stage (image to markdown)
main_lm = OpenRouterClient(
    model=OCR_MODEL,
    temperature=1.0,
    max_tokens=100000,
    reasoning={"enabled": True},
)

# Reflection model used by GEPA to analyze errors and suggest improvements
reflection_lm = OpenRouterClient(
    model=REFLECTION_MODEL,
    temperature=1.0,
    max_tokens=32000,
    reasoning_effort="high",
)

# Configure DSPy to use our main language model
dspy.configure(lm=main_lm, adapter=dspy.ChatAdapter())


########################################
# STAGE 1: Image → Markdown
########################################

class FreeformVisionAdapter(Adapter):
    """Custom DSPy adapter for vision models that handles image inputs.

    Formats the prompt with both text instructions and the image URL,
    then parses the raw model output as markdown without enforcing structure.
    """

    def format(self, signature: type[Signature], demos, inputs):
        """Format the input for the vision model with image and text."""
        img = inputs.get("image")
        user_content = [
            {"type": "text", "text": "Convert this document image to Markdown."}
        ]
        if getattr(img, "url", None):
            user_content.append({"type": "image_url", "image_url": {"url": img.url}})
        return [
            {"role": "system", "content": signature.instructions},
            {"role": "user", "content": user_content},
        ]

    def parse(self, signature: type[Signature], completion: str) -> dict:
        """Parse the model output - just return the raw markdown."""
        return {"markdown_output": completion}


class ImageToMarkdownSignature(dspy.Signature):
    """
    DSPy signature defining the task of converting a document image to markdown.

    This is the base instruction that GEPA will optimize. The rules below define
    the expected markdown format for document elements like tables, logos, and checkboxes.

    Convert the following document to markdown.
    Return only the markdown with no explanation text.

    RULES:
        - You must include all information on the page. Do not exclude headers, footers, charts, infographics, or subtext.
        - Return tables in an HTML format.
        - Logos should be wrapped in brackets. Ex: <logo>Coca-Cola<logo>
        - Watermarks should be wrapped in brackets. Ex: <watermark>OFFICIAL COPY<watermark>
        - Page numbers should be wrapped in brackets. Ex: <page_number>14<page_number> or <page_number>9/22<page_number>
        - Prefer using ☐ and ☑ for check boxes.

    """

    image: dspy.Image = dspy.InputField()
    markdown_output: str = dspy.OutputField(desc="Formatted Markdown string")


class ImageToMarkdown(dspy.Module):
    """DSPy module that converts document images to markdown using a vision model."""

    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(ImageToMarkdownSignature)

    def forward(self, image: dspy.Image, **_ignored) -> dspy.Prediction:
        """Run the image through the vision model to extract markdown."""
        with dspy.context(adapter=FreeformVisionAdapter()):
            raw = self.predict(image=image)

        return raw


########################################
# STAGE 2: Markdown → JSON
########################################

class MarkdownToJsonSignature(dspy.Signature):
    """Extract structured data from markdown content based on a JSON schema.

    You are a specialized data extraction system. Your task is to analyze the provided markdown
    document and extract relevant information according to the given JSON schema structure.

    RULES:
    - Extract only information that is explicitly present in the markdown
    - Follow the exact structure and data types specified in the schema
    - Return null/empty values for fields not present in the document
    - Preserve exact values, spellings, and formatting from the source
    - For dates, amounts, and structured data, extract precisely as shown
    - Do not infer or hallucinate information not explicitly stated

    Return only valid JSON that conforms to the provided schema.
    """

    markdown: str = dspy.InputField(desc="Markdown content to extract from")
    json_schema: str = dspy.InputField(
        desc="JSON schema definition for extraction structure"
    )
    json_output: str = dspy.OutputField(desc="Extracted JSON data conforming to schema")


class MarkdownToJson(dspy.Module):
    """DSPy module that extracts structured JSON from markdown based on a schema."""

    def __init__(self):
        super().__init__()

    def forward(self, markdown: str, json_schema: str | dict) -> dspy.Prediction:
        """Extract JSON data from markdown according to the provided schema.

        Uses an Agno agent with chain-of-thought reasoning to parse markdown
        and extract data into a Pydantic model generated from the JSON schema.
        """
        # Normalize schema to string for prompt clarity
        try:
            schema_str = (
                json.dumps(json_schema, indent=2, ensure_ascii=False)
                if isinstance(json_schema, (dict, list))
                else str(json_schema)
            )
        except Exception:
            schema_str = str(json_schema)

        instructions = f"""
You are an expert at json extraction from markdown.

Extract structured data from markdown content based on the JSON schema below.

Rules:
- Extract only information explicitly present in the markdown (no hallucinations)
- Follow the exact structure and data types in the schema
- Use null/empty values for missing fields
- Preserve exact values, spellings, numbers, units, and punctuation
- Return only valid JSON conforming to the schema

EXTREMELY IMPORTANT!!
**Required JSON Schema:**
{schema_str}


"""

        # Generate a Pydantic model from the JSON schema for structured output
        response_model = generate_pydantic_model("JsonExtraction", json_schema)

        # Create an Agno agent that uses chain-of-thought reasoning
        json_agent = Agent(
            name="json_agent",
            instructions=instructions,
            success_criteria="You have thought through the extraction and have within <response> tags placed a valid json object that matches the schema.",
            model=OpenRouter(
                id=JSON_MODEL,
                api_key=os.environ["OPENROUTER_API_KEY"],
                max_tokens=32000,
                temperature=0.0,
            ),
        )

        message = f"Here is the markdown to convert to JSON according to the schema.\n\n{markdown}"
        json_output = run_custom_structured_output_agent_with_chain_of_thought(
            agent=json_agent,
            response_model=response_model,
            message=message,
        )

        final_json_output = json_output.model_dump(mode="json")

        return dspy.Prediction(json_output=final_json_output)


########################################
# COMBINED PIPELINE
########################################

class ImageToJsonPipeline(dspy.Module):
    """Full pipeline that converts a document image to structured JSON.

    Chains together the two stages:
      1. Image → Markdown (vision model)
      2. Markdown → JSON (language model with schema)
    """

    def __init__(self):
        super().__init__()
        self.markdown_extractor = ImageToMarkdown()
        self.json_extractor = MarkdownToJson()

    def forward(self, image: dspy.Image, json_schema: str) -> dspy.Prediction:
        """Run the full pipeline on a document image."""
        # Step 1: Extract markdown from image
        markdown_result = self.markdown_extractor(image=image)

        # Step 2: Extract JSON from markdown using the provided schema
        json_result = self.json_extractor(
            markdown=markdown_result.markdown_output, json_schema=json_schema
        )

        return dspy.Prediction(
            markdown_output=markdown_result.markdown_output,
            json_output=json_result.json_output,
        )


########################################
# DATA LOADING
########################################

def get_image_hash(image_url: str) -> str:
    """Generate MD5 hash for image URL (matches TypeScript implementation)."""
    return hashlib.md5(image_url.encode()).hexdigest()


def extract_after_test(url: str, segment: str = "test"):
    """Extract the segment after 'test' in a URL path."""
    path = urlparse(url).path
    parts = [p for p in path.split("/") if p]
    try:
        i = parts.index(segment)
        return parts[i + 1] if i + 1 < len(parts) else None
    except ValueError:
        return None


def load_example_from_json(json_path: Path, img_path: Path):
    """Load a document example from JSON file and corresponding image.

    Converts the image to a base64 data URL for the vision model and
    extracts ground truth markdown and JSON for evaluation.
    """
    # Load JSON data
    with open(json_path) as f:
        data = json.load(f)

    # Convert image to base64 data URL for vision model
    with open(img_path, "rb") as f:
        img_data = f.read()

    # Determine image format from extension
    ext = img_path.suffix.lower()
    if ext in [".jpg", ".jpeg"]:
        mime_type = "image/jpeg"
    elif ext == ".png":
        mime_type = "image/png"
    elif ext == ".gif":
        mime_type = "image/gif"
    elif ext == ".webp":
        mime_type = "image/webp"
    else:
        mime_type = "image/jpeg"

    b64_img = base64.b64encode(img_data).decode("utf-8")
    data_url = f"data:{mime_type};base64,{b64_img}"

    # Extract ground truth data for evaluation
    true_markdown = data.get("trueMarkdownOutput", "")
    true_json = data.get("trueJsonOutput", "")

    return dspy.Example(
        image=dspy.Image.from_url(data_url),
        image_index=extract_after_test(data.get("imageUrl", "")),
        image_url=data.get("imageUrl", ""),
        true_markdown=true_markdown,
        metadata=data.get("metadata", {}),
        json_schema=data.get("jsonSchema", {}),
        doc_index=int(json_path.stem),
        true_json=true_json,
    ).with_inputs("image", "json_schema")


def init_dataset(limit: int = None):
    """Initialize dataset from downloaded JSON files and cached images.

    Args:
        limit: Maximum number of examples to load (None = load all)

    Returns:
        List of dspy.Example objects with images, schemas, and ground truth

    Raises:
        FileNotFoundError: If data or images are missing (with helpful message)
    """
    # Get all available document pairs from ./data and ./image-cache
    pairs = list_all_document_pairs()

    if not pairs:
        raise FileNotFoundError(
            "No JSON files found in ./data. Run `python download_data.py` first to download the benchmark."
        )

    # Check for missing files and provide helpful error messages
    missing_json = [idx for idx, json_path, _ in pairs if json_path is None]
    if missing_json:
        raise FileNotFoundError(
            "JSON files are missing for some document indices. Re-run `python download_data.py` to rebuild ./data."
        )

    missing_images = [idx for idx, _, img_path in pairs if img_path is None]
    if missing_images:
        raise FileNotFoundError(
            "Missing cached images for the dataset. Run `python download_data.py` (without --no-images) to populate ./image-cache."
        )

    # Filter to only complete pairs (both JSON and image exist)
    complete_pairs = [
        (idx, json_path, img_path)
        for idx, json_path, img_path in pairs
        if json_path and img_path
    ]

    print(f"Found {len(complete_pairs)} complete document pairs")

    # Apply limit if specified
    if limit:
        complete_pairs = complete_pairs[:limit]
        print(f"Using first {len(complete_pairs)} documents")

    # Load examples into DSPy format
    data = []
    for idx, json_path, img_path in complete_pairs:
        try:
            example = load_example_from_json(json_path, img_path)
            data.append(example)
        except Exception as e:
            print(f"Warning: Failed to load document {idx}: {e}")
            continue

    print(f"Successfully loaded {len(data)} examples")
    return data


########################################
# EVALUATION METRICS
########################################

def metric_with_feedback(
    example, prediction, trace=None, pred_name=None, pred_trace=None
):
    """Deterministic metric for GEPA optimization.

    Compares predicted JSON to ground truth using the TypeScript scorer
    and provides detailed feedback for the GEPA reflection model to analyze.

    Returns:
        dspy.Prediction with:
          - score: float in [0,1] from TypeScript evaluation
          - feedback: detailed comparison of predicted vs ground truth
    """
    # Parse ground-truth JSON
    try:
        actual_json = (
            json.loads(example.true_json)
            if isinstance(example.true_json, str)
            else example.true_json
        )
    except Exception:
        actual_json = example.true_json if isinstance(example.true_json, dict) else {}

    # Parse predicted JSON from program output
    pred_json_field = getattr(prediction, "json_output", {})
    try:
        predicted_json = (
            json.loads(pred_json_field)
            if isinstance(pred_json_field, str)
            else pred_json_field
        )
    except Exception:
        predicted_json = pred_json_field if isinstance(pred_json_field, dict) else {}

    # Compute deterministic score using the TypeScript evaluation script
    score = call_typescript_json_evaluation(actual_json, predicted_json)

    # Build detailed feedback for GEPA to understand what went wrong
    pred_markdown = getattr(prediction, "markdown_output", "") or ""
    gt_markdown = getattr(example, "true_markdown", "") or ""

    try:
        predicted_json_text = json.dumps(predicted_json, indent=2, ensure_ascii=False)
    except Exception:
        predicted_json_text = str(predicted_json)
    try:
        actual_json_text = json.dumps(actual_json, indent=2, ensure_ascii=False)
    except Exception:
        actual_json_text = str(actual_json)

    feedback_text = (
        "Results from OCR pipeline.\n"
        f"Score: {score:.3f}\n\n"
        # MARKDOWN STAGE
        "MARKDOWN STAGE:\n"
        f"Predicted Markdown:\n{pred_markdown}\n\n"
        f"Ground Truth Markdown:\n{gt_markdown}\n\n"
        # JSON EXTRACTION STAGE
        "JSON EXTRACTION STAGE:\n"
        f"Predicted JSON:\n{predicted_json_text}\n\n"
        f"Ground Truth JSON:\n{actual_json_text}\n\n"
        # Pipeline context
        f"Pipeline context:\nWe use a two step pipeline that first uses ocr to produce the markdown, and then another process that extracts from the markdown to produce json."
    )

    return dspy.Prediction(score=score, feedback=feedback_text)


########################################
# MAIN ENTRY POINT
########################################

def main(train: bool = True, version: str = "v1"):
    """Main function to run GEPA optimization or evaluation.

    Args:
        train: If True, run GEPA optimization. If False, load and evaluate saved program.
        version: Version identifier for saving/loading optimized programs
    """
    # Load the dataset
    data = init_dataset()
    current_program = ImageToJsonPipeline()

    # Shuffle with fixed seed for reproducibility
    random.seed(42)
    random.shuffle(data)

    if train:
        # Split into validation and training sets
        val_n = 30
        val_set = data[:val_n]
        train_set = data[val_n:]

        # Configuration for experiment tracking
        experiment_config = {
            "reflection_lm": REFLECTION_MODEL_ID,
            "ocr_model": OCR_MODEL_ID,
            "json_model": OR_JSON_MODEL,
            "train_set_size": len(train_set),
            "val_set_size": len(val_set),
            "dataset": "full_sample",
            "experiment_goal": "auto",
            "temperature": 1.0,
            "reasoning": True,
            "version": version,
        }

        # Weights & Biases logging configuration
        wandb_init_kwargs = {
            "project": "ocr-benchmarking",
            "name": f"gepa-v{version}",
            "group": "ocr-experiments",
            "tags": ["gepa", "baseline", experiment_config["dataset"]],
            "notes": f"GEPA optimization with {experiment_config['train_set_size']} train set and {experiment_config['val_set_size']} val set temperature 1.0",
            "config": experiment_config,
        }

        # Create logging directory
        log_dir = Path(f"./v{version}")
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / "prog_candidates").mkdir(exist_ok=True)

        # Initialize GEPA optimizer
        optimizer = GEPA(
            metric=metric_with_feedback,
            auto="light",  # Light mode for faster optimization
            num_threads=32,  # Parallel evaluation
            include_images_in_reflection=False,  # Don't send images to reflection model
            use_merge=True,  # Merge improvements from multiple candidates
            reflection_lm=reflection_lm,
            wandb_init_kwargs=wandb_init_kwargs,
            track_stats=True,
            track_best_outputs=True,
            use_wandb=True,
            wandb_api_key=os.environ["WANDB_API_KEY"],
            skip_perfect_score=True,  # Skip examples with perfect scores
            log_dir=f"./v{version}",
        )

        # Run optimization
        optimized_program = optimizer.compile(
            current_program,
            trainset=train_set,
            valset=val_set,
        )
        prog_name = f"optimized_programv{version}"

        # Save optimized program
        optimized_program.save(
            f"{prog_name}.json",
            save_program=False,
        )
    else:
        # Load previously optimized program
        current_program.load(f"optimized_programv{version}.json")

        # Evaluate on full dataset
        evaluate = dspy.Evaluate(
            devset=data,
            metric=deterministic_json_metric,
            num_threads=32,
            display_table=True,
            display_progress=True,
        )

        result = evaluate(current_program)
        print(result)


if __name__ == "__main__":
    main()
