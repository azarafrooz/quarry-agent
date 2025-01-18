from ..utils import GeneratedTool
from jinja2 import Template
import importlib.resources as resources
import re
import json
from ..exceptions import LLMResponseJsonFormatError
from typing import List, Tuple
from io import StringIO
import csv
from typing import Optional
import logging

# Setup logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

ACTION_DESCRIPTION_TEMPLATE = Template(
    "- {{ tool.name }}({{ tool.inputs }}) -> {{ tool.output_type }}: {{ tool.description }}"
)


def format_action(tool: GeneratedTool) -> str:
    return f"- {tool.name}({tool.inputs}) -> {tool.output_type}: {tool.description}"


def format_action_jinja(tool: GeneratedTool) -> str:
    return ACTION_DESCRIPTION_TEMPLATE.render(tool=tool)


# and `resources` with the directory containing the .md file.
def read_resource_file(filename):
    with resources.files("quarry.resources").joinpath(filename).open("r") as f:
        return f.read()


REACT_PROMPT = read_resource_file("react_prompt.md")


REWARD_PROMPT = read_resource_file("reward_prompt.md")


def extract_python_codes(markdown_string: str) -> list[str]:
    """
    Extract all Python code blocks from a markdown string.

    Args:
        markdown_string (str): The input markdown string.

    Returns:
        list: A list of Python code blocks, or an empty list if none are found.
    """
    # Regex pattern to match Python code blocks
    pattern = r"```[\w\s]*python\n([\s\S]*?)```"

    # Find all matches in the markdown string
    matches = re.findall(pattern, markdown_string, re.IGNORECASE)

    # Clean up the extracted code blocks
    python_codes = [match.strip() for match in matches]

    return python_codes


def trim_after_first_python_code_block(text):
    start_index = text.find("```python")
    if start_index == -1:
        return text  # No "```python" found, return the text as is

    end_index = text.find("```", start_index + len("```python"))
    if end_index == -1:
        return text[
            : start_index + len("```python")
        ].strip()  # No closing ``` found, return until opening ```python

    return text[: end_index + len("```")].strip()  # Include the closing ```


def extract_and_validate_json(text):
    # Regular expression to extract JSON objects with "thought" and "code" keys
    json_pattern = re.compile(
        r"""
        \{
            \s*"thought"\s*:\s*".*?"\s*,\s*  # Match "thought" key with its value
            "code"\s*:\s*".*?"\s*           # Match "code" key with its value
        \}
    """,
        re.VERBOSE | re.DOTALL,
    )

    matches = json_pattern.findall(text)

    if not matches:
        raise LLMResponseJsonFormatError(
            "No valid JSON objects found in the LLM response."
        )

    parsed_objects = []
    for match in matches:
        try:
            # Parse and validate JSON
            json_obj = json.loads(match)
            parsed_objects.append(json_obj)
        except json.JSONDecodeError as e:
            raise LLMResponseJsonFormatError(
                "JSON found in LLM response is not in the expected format"
            ) from e

    return parsed_objects


def append_kernel_output(action, kernel_output):
    """Append the interpreter outcome. Also append thought to avoid code result hallucinations
    Limit the max Kernel output to 10K (approx 2.5words)
    """
    return action + f"""\nInterpreter trace:\n{kernel_output[:10000]}\n"thought":"""


def is_action_complete(action: str) -> bool:
    return True if "submit_final_answer" in action else False


def format_reward_prompt(analysis: str) -> str:
    findings = analysis.split("## Goal")[1]
    return REWARD_PROMPT.format(findings=findings)


def extract_score(llm_output: str) -> Optional[float]:
    """
    Extract the score from LLM output, accepting both "score is N" and "score=N" formats
    where N is a float between -1 and 1.

    Args:
        llm_output (str): Raw output string from the LLM

    Returns:
        float or None: Extracted score if valid, None if no valid score found

    Examples:
        >>> extract_score("score is 0.75")
        0.75
        >>> extract_score("Score=-0.5")  # Case insensitive
        -0.5
        >>> extract_score("the score = 0.9")  # Handles extra spaces
        0.9
        >>> extract_score("score=0.8")  # Equals format
        0.8
        >>> extract_score("invalid")  # Invalid format
        -1.0
    """
    # Match case-insensitive "score" followed by either:
    # - "is" or "=" with optional spaces
    # - negative sign, digits, optional decimal point and more digits
    pattern = r"(?i)score\s*(?:is|=)\s*(-?\d*\.?\d+)"

    match = re.search(pattern, llm_output)
    if not match:
        return -1.0  # None
    try:
        score = float(match.group(1))
        # Validate score is in range [-1, 1]
        if score < -1 or score > 1:
            return -1.0  # None
        # Round to 2 decimal places
        return round(score, 2)
    except ValueError:
        return -1.0  # None


# def extract_integer(text: str) -> int:
#     match = re.search(r"\bscore\s*(?:is|:)\s*(\d+)", text, re.IGNORECASE)
#     if match:
#         score = int(match.group(1))
#     else:
#         score = 0
#     return score


DOCUMENT_CONTEXT_PROMPT = """
<document>
{doc_content}
</document>
"""

CHUNK_CONTEXT_PROMPT = """
Here is the chunk we want to situate within the whole document
<chunk>
{chunk_content}
</chunk>

Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk.
Answer only with the succinct context and nothing else.
"""


# def parse_csv_content(text: str) -> List[Tuple[str, List[dict]]]:
#     """
#     Parse markdown content containing CSV code blocks and return a list of tuples
#     containing (filename, parsed_csv_content).
#
#     Args:
#         text (str): Input text containing markdown and CSV code blocks
#
#     Returns:
#         List[Tuple[str, List[dict]]]: List of (filename, csv_content) tuples
#     """
#     # Initialize results list
#     results = []
#
#     try:
#         # Pattern to match CSV code blocks
#         # Captures filename from ```filename.csv and the content until the next ```
#         pattern = r"```(\w+\.csv)\n(.*?)```"
#
#         # Find all matches in the text
#         matches = re.finditer(pattern, text, re.DOTALL)
#
#         for match in matches:
#             filename = match.group(1)
#             csv_content = match.group(2).strip()
#             try:
#                 # Parse CSV content using StringIO
#                 csv_reader = csv.DictReader(StringIO(csv_content))
#                 results.append((filename, csv_content))
#                 # logger.info(f"Successfully parsed CSV content for {filename}")
#
#             except csv.Error as e:
#                 # logger.error(f"Error parsing CSV content for {filename}: {str(e)}")
#                 continue
#
#     except Exception as e:
#         # logger.error(f"Unexpected error during parsing: {str(e)}")
#         raise
#
#     return results


# REACT_PROMPT = """# Instructions
# You are an AI assistant that helps users derive actionable information from raw database data. You have access to a Python interpreter with access to <<dialect>> database with global env `conn'.
#
# When given a task, proceed step by step to solve it. At each step:
# 1. **Thought**: Briefly explain your reasoning and what you plan to do next.
# 2. **Code**: Provide Python code that implements your plan. To interact with or gather information from database assume the conncetion variable is `conn'. To process or read data further, feel free to use `panda', `numpy', `scikit-learn' etc.
# The interpreter will execute your code and return the results to you. Review the results from current and previous steps to decide your next action.
#
# **Continue this process until you find the solution or reach a maximum of <<max_iterations>> iterations.** Once you have the final answer, use the `submit_final_answer` function to return it to the user.
#
# # Output Format
# At each step, output a JSON object in the following format:
#
# ```json
# {
#     "thought": "Your thought here.",
#     "code": "Your Python code here."
# }
# ```
#
# # Available Functions
# You are provided with several available functions. If you need to discover more relevant functions, use the `get_relevant_tools` function.
# ```
# <<tool_descriptions>>
# ```
#
# # Guidelines for Writing Code
# 1. First, decide whether to reuse an existing function or define a new one.
# 2. Look at the list of available functions. If no existing function is relevant, run `get_relevant_tools` to find more functions and proceed to the next step.
# 3. If the retrieved functions are still not relevant, define a new function.
# 4. When implementing a new function, you must ensure the following:
#    - The function is abstract, modular, and reusable. Specifically, the function name must be generic (e.g., `count_objects` instead of `count_apples`). The function must use parameters instead of hard-coded values specially in the SQL so it will be reusable for other related tasks. The function body must be self-contained.
#    - Explicitly declare input and output data types using type hints.
#    *Example*: `def function_name(param: int) -> str:`
#    - Include a one-line docstring describing the function's purpose, following PEP 257 standards.
#    - When your function calls multiple other functions that are not from a third-party library, ensure you print the output after each call. This will help identify any function that produces incorrect or unexpected results.
#
# # Guidelines for Analyzing the Output
# After execution, analyze the output as follows:
# 1. If the code fails to execute successfully and an error is returned, read the error message and traceback carefully, then revise your code in the next step.
# 2. If the code executes successfully and an output is returned, proceed as follows:
#    - If the output contains relevant information, you can move on to the next step.
#    - If the output does not contain any relevant information, consider alternative approaches. For example, try different sql queries, use different functions or libraries, implement new functions if necessary.
#
# # Important Notes
# 1. When querying from database make sure you don't miss any details and arrive at the wrong conclusion. And yet be cautious that data could be very large. So use Limit or head operations to limit when appropriate.
# 2. Base your decisions on database data. Rely solely on this data to generate your answers; do not rely on your own knowledge, and do not imagine data out of nowhere, as it will mislead you to an incorrect answer. In your code, write comments that cite your data source (e.g., which row in the result, etc.) so that a human can verify them.
# 3. DO NOT GIVE UP. Keep trying until you reach the maximum iteration limit.
#
# # Context:
# <<context>
#
# Goal: <<goal>>
# Begin.
# """
