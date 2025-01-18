from typing import List, Dict, NamedTuple
from pydantic import BaseModel
from dataclasses import dataclass


@dataclass
class GeneratedTool:
    """
    Represents a modular function extracted from code.
    """

    name: str
    description: str
    inputs: str
    output_type: str
    code: str
    dependencies: str


# Define the Pydantic models to match the function's JSON structure
class FunctionParameter(BaseModel):
    type: str


class FunctionParameters(BaseModel):
    type: str = "object"
    properties: Dict[str, FunctionParameter]
    required: List[str]


class FunctionSchema(BaseModel):
    name: str
    description: str
    parameters: FunctionParameters


class FunctionModel(BaseModel):
    type: str = "function"
    function: FunctionSchema


class ReActState(NamedTuple):
    """The state of the ReAct."""

    step_idx: int
    prompt: str
    action: str
    goal: str
    finished: bool
