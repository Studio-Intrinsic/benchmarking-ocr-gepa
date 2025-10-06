"""
Parsing Utilities for Chain-of-Thought Responses

Utilities for parsing structured outputs from language models that use
chain-of-thought reasoning. Handles <think> and <response> tags, JSON extraction,
and dynamic Pydantic model generation from JSON schemas.
"""

from typing import Dict, Type, Tuple, Any, Optional, List
import re
import json
from pydantic import BaseModel, Field, create_model
from agno.agent import Agent


DEFAULT_TENACITY_RETRY_ATTEMPTS = 3

# Regex patterns to extract JSON from <response> tags (tries multiple formats)
DEFAULT_RESPONSE_PATTERNS = [
    # JSON code fence inside <response>
    r"<response[^>]*>\s*```\s*(?:json|JSON|Json)\s*(.*?)\s*```\s*</response>",
    # Bare JSON object/array inside <response>
    r"<response[^>]*>\s*(\{[\s\S]*?\})\s*</response>",
    r"<response[^>]*>\s*(\[[\s\S]*?\])\s*</response>",
    # Fallback: any <response> content (we will extract inner JSON later)
    r"<response[^>]*>\s*([\s\S]*?)\s*</response>",
    # Code fence with json (outside of response) as a last resort
    r"```\s*(?:json|JSON|Json)\s*(.*?)\s*```",
]

DEFAULT_REASONING_PATTERNS = [
    r"<think>\s*(.*?)\s*</think>",
    r"^(.*?)</think>",
]

MARKDOWN_RE = re.compile(r"<final_markdown>([\s\S]*?)</final_markdown>", re.I)


def run_custom_structured_output_agent_with_chain_of_thought(
    agent: Agent,
    response_model: Type[BaseModel],
    message: str,
    allow_none: bool = False,
) -> Type[BaseModel] | None:
    """Run an Agno agent with chain-of-thought reasoning and structured output.

    The agent is instructed to use <think> tags for reasoning and <response> tags
    for the final JSON output. Retries up to 3 times if parsing fails.

    Args:
        agent: Agno agent to use
        response_model: Pydantic model defining the expected output structure
        message: User message to send to the agent
        allow_none: Whether to allow null as a valid response

    Returns:
        Parsed Pydantic model instance, or None if allow_none=True and model returned null
    """
    error_context = ""

    for attempt in range(DEFAULT_TENACITY_RETRY_ATTEMPTS):
        none_clause = (
            "\n- If there is no valid result, return a literal JSON null inside <response> (i.e., null)"
            if allow_none
            else ""
        )

        agent.instructions += f"""

You must analyze the given input and respond using a specific format with two sections:

1. **Think**: Use <think></think> tags to work through your reasoning step by step
2. **Response**: Use <response></response> tags to provide valid JSON that matches the required schema

**Output Format:**
<think>
[Your step-by-step analysis and reasoning here]
</think>
<response>
[Valid JSON object that conforms to the schema below]
</response>

EXTREMELY IMPORTANT!!
**Required JSON Schema (formal Pydantic schema):**
{response_model.model_json_schema()}

**Critical Requirements:**
- The JSON in <response> tags must be valid and parseable
- The JSON must conform exactly to the provided schema
- Include all required fields from the schema
- Use appropriate data types (strings, numbers, booleans, arrays, objects)
{none_clause}

{error_context}

"""

        try:
            response = agent.run(message)
            json_response, _ = parse_chain_of_thought_to_json(response.content)
            # Allow explicit null to represent no result
            if allow_none and json_response is None:
                return None
            return response_model(**json_response)
        except Exception:
            if attempt < DEFAULT_TENACITY_RETRY_ATTEMPTS - 1:
                error_context = f"""
**IMPORTANT - Previous Attempt Failed:**
Your previous response failed to produce valid JSON. Common issues:
- Missing or malformed <response> tags
- Invalid JSON syntax (missing quotes, trailing commas, etc.)
- JSON that doesn't match the required schema
- Extra text outside the JSON object

Please ensure your JSON is properly formatted and valid.{" If there is no valid result, you may return a literal JSON null." if allow_none else ""}
"""
                # Reset instructions for next attempt
                agent.instructions = agent.instructions.split("You must analyze")[0]
            else:
                raise

    return None


def parse_chain_of_thought_to_json(
    text: str, include_reasoning: bool = True
) -> Tuple[dict[str, Any], Optional[str]]:
    """Parse JSON from chain-of-thought text with <response> tags.

    Tries multiple regex patterns to extract JSON from <response> blocks.
    Also extracts reasoning from <think> blocks if requested.

    Returns:
        Tuple of (parsed_json_dict, reasoning_text)
    """
    if include_reasoning:
        reasoning = parse_chain_of_thought_to_reasoning(text)
    else:
        reasoning = None

    def _sanitize(s: str) -> str:
        """Remove invisible Unicode characters that can break JSON parsing."""
        # Remove BOM and zero-width / non-breaking spaces that can break json.loads
        invisible_chars = [
            "\ufeff",  # BOM
            "\u200b",  # zero-width space
            "\u200c",  # zero-width non-joiner
            "\u200d",  # zero-width joiner
            "\u2060",  # word joiner
            "\xa0",  # non-breaking space
        ]
        for ch in invisible_chars:
            s = s.replace(ch, "")
        return s.strip()

    for pattern in DEFAULT_RESPONSE_PATTERNS:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            candidate = _sanitize(match.group(1))
            # If the model explicitly returned JSON null
            if candidate == "null":
                return None, reasoning  # type: ignore[return-value]
            # Quick guard: only attempt json if plausible start
            if not candidate.startswith("{") and not candidate.startswith("["):
                # Attempt to extract the first JSON object/array within the candidate
                inner = re.search(r"(\{[\s\S]*\})", candidate)
                if not inner:
                    inner = re.search(r"(\[[\s\S]*\])", candidate)
                if inner:
                    candidate = _sanitize(inner.group(1))
                else:
                    print(
                        f"Skipping candidate that doesn't look like JSON: {candidate[:80]}"
                    )
                    continue
            try:
                return json.loads(candidate), reasoning
            except json.JSONDecodeError as e:
                print(f"Failed to decode JSON: {e}")
                # Try next pattern instead of failing immediately
                continue
    raise Exception("Failed to decode JSON")


def parse_chain_of_thought_to_boolean(
    text: str, include_reasoning: bool = True
) -> Tuple[bool, Optional[str]]:

    if include_reasoning:
        reasoning = parse_chain_of_thought_to_reasoning(text)
    else:
        reasoning = None

    for pattern in DEFAULT_RESPONSE_PATTERNS:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return match.group(1).strip() == "true", reasoning
            except Exception as e:
                raise Exception("Failed to decode JSON")
    raise Exception("Failed to decode JSON")


def parse_chain_of_thought_to_reasoning(text: str) -> Optional[str]:
    try:
        for pattern in DEFAULT_REASONING_PATTERNS:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                return match.group(1).strip()
        return None
    except Exception as e:
        return None


def extract_markdown_and_clean(raw_response: str) -> str:
    if not raw_response:
        return ""
    # Find all complete pairs and take the last one.
    matches = list(MARKDOWN_RE.finditer(raw_response))
    if matches:
        return matches[-1].group(1).strip()
    # Fallback: nothing matched—return the whole blob trimmed.
    res = raw_response.strip()
    # remove any ``` or markdown
    res = re.sub(r"```", "", res)
    res = re.sub(r"markdown", "", res)
    return res


def generate_pydantic_model(name: str, schema_def: Dict[str, Any]) -> type[BaseModel]:
    """Dynamically generate a Pydantic model from a JSON schema definition.

    Supports:
      - Primitive types (string, number, integer, boolean)
      - Enums
      - Nested objects and arrays
      - Field descriptions

    Args:
        name: Name for the generated Pydantic model
        schema_def: JSON schema definition (must have "properties" key)

    Returns:
        Dynamically created Pydantic model class
    """
    properties: Dict[str, tuple[Any, Any]] = {}

    for key, value in schema_def.get("properties", {}).items():
        field_type = None

        if (
            "enum" in value
            and isinstance(value["enum"], list)
            and len(value["enum"]) > 0
        ):
            # Enums
            from enum import Enum

            enum_name = f"{name}_{key.capitalize()}Enum"
            field_type = Enum(enum_name, {str(v): v for v in value["enum"]})
        else:
            # Map JSON schema types to Python
            type_mapping = {
                "array": List[Any],
                "boolean": bool,
                "integer": int,
                "number": float,
                "string": str,
                "object": dict,
            }
            field_type = type_mapping.get(value["type"], Any)

        # Handle nested arrays / objects
        if value["type"] == "array" and "items" in value:
            if value["items"]["type"] == "object":
                nested_model = generate_pydantic_model(
                    f"{name}_{key.capitalize()}", value["items"]
                )
                field_type = List[nested_model]
            else:
                inner_type = {
                    "boolean": bool,
                    "integer": int,
                    "number": float,
                    "string": str,
                    "object": dict,
                }.get(value["items"]["type"], Any)
                field_type = List[inner_type]

        elif value["type"] == "object":
            nested_model = generate_pydantic_model(f"{name}_{key.capitalize()}", value)
            field_type = nested_model

        # Make fields nullable by default
        field_type = Optional[field_type]

        # Add field description if available
        field_args = {}
        if "description" in value:
            field_args["description"] = value["description"]

        properties[key] = (field_type, Field(None, **field_args))

    return create_model(name, **properties)  # dynamic Pydantic model
