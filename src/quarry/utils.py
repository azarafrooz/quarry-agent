import ast
import os
from glob import glob
from typing import Union, List, Tuple, Set
import jsonref
import hashlib
import inspect
import uuid
import sqlparse
import builtins
import importlib
from quarry.types import GeneratedTool, FunctionModel
from quarry.exceptions import InterpreterEnvUpdateError

from functools import wraps
from typing import Callable, Dict, Any


class CountCalls:
    def __init__(self):
        self.count = 0

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            self.count += 1
            return func(*args, **kwargs)

        return wrapper


def extract_description(func_def: ast.FunctionDef) -> str:
    """
    Extract the first line of the docstring of a function.
    """
    docstring = ast.get_docstring(func_def)
    return docstring.strip().split("\n")[0] if docstring else "No description provided"


def extract_inputs(func_def: ast.FunctionDef) -> str:
    """
    Extract function inputs as a string.
    """
    return ast.unparse(func_def.args) if func_def.args else ""


def extract_output_type(func_def: ast.FunctionDef) -> str:
    """
    Extract the return type annotation of a function.
    """
    return ast.unparse(func_def.returns) if func_def.returns else ""


def extract_func_info(func_def: ast.FunctionDef) -> dict:
    """
    Extract detailed information from a function definition.
    """
    return {
        "name": func_def.name,
        "description": extract_description(func_def),
        "inputs": extract_inputs(func_def),
        "output_type": extract_output_type(func_def),
        "code": ast.unparse(func_def),
    }


def add_parent_pointers(node: ast.AST) -> None:
    """
    Add parent pointers to nodes in an AST for easier traversal.
    """
    for child in ast.iter_child_nodes(node):
        child.parent = node  # type: ignore[attr-defined]
        add_parent_pointers(child)


def parse_functions_from_code(code: str) -> List[GeneratedTool]:
    """
    Parse LLM-generated code to extract functions, descriptions and dependencies,
    including imports with aliases. Handles edge cases with string literals and newlines.
    """
    # Preprocess the code to handle edge cases
    try:
        # First attempt: try parsing the original code
        tree = ast.parse(code)
    except SyntaxError:
        # If parsing fails, try to clean up the code
        lines = code.split("\n")
        cleaned_lines = []
        in_function = False

        for line in lines:
            # Keep track of whether we're inside a function definition
            if line.strip().startswith("def "):
                in_function = True
            # Only include lines that are part of function definitions or imports
            if in_function or line.strip().startswith(("import ", "from ")):
                cleaned_lines.append(line)
            # Check for end of function
            if in_function and line.strip() == "":
                in_function = False

        cleaned_code = "\n".join(cleaned_lines)
        tree = ast.parse(cleaned_code)

    add_parent_pointers(tree)

    import_statements = []
    func_defs = []

    # Collect imports and top-level functions
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:
                    import_statements.append(f"import {alias.name} as {alias.asname}")
                else:
                    import_statements.append(f"import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                if alias.asname:
                    import_statements.append(
                        f"from {module} import {alias.name} as {alias.asname}"
                    )
                else:
                    import_statements.append(f"from {module} import {alias.name}")
        elif isinstance(node, ast.FunctionDef):
            func_defs.append(node)

    dependencies = "\n".join(import_statements)
    generated_tools = [
        GeneratedTool(**extract_func_info(func_def), dependencies=dependencies)
        for func_def in func_defs
    ]

    return generated_tools


def get_action_set(res_path: str) -> Set[str]:
    """
    Retrieve the action set from generated scripts.
    """
    agent_name = os.path.basename(res_path).split(".")[0]
    generated_tool_dir = os.path.join("../generated_tools", agent_name)

    action_set = set()
    for path in sorted(glob(os.path.join(generated_tool_dir, "*.py"))):
        with open(path, "r") as file:
            tools = parse_functions_from_code(file.read())
            if tools:
                action_set.add(tools[0].name)
    return action_set


def extract_function_calls(code: str) -> List[str]:
    """
    Extract function calls from code.
    """
    tree = ast.parse(code)
    calls = []

    class FunctionCallVisitor(ast.NodeVisitor):
        def visit_Call(self, node):
            if isinstance(node.func, ast.Name):
                calls.append(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                calls.append(node.func.attr)
            self.generic_visit(node)

    FunctionCallVisitor().visit(tree)
    return calls


def is_sufficient(code: str, action_set: Set[str]) -> Union[bool, None]:
    """
    Determine if the code uses only functions from the action set.
    """
    try:
        function_calls = extract_function_calls(code)
        return all(func_call in action_set for func_call in function_calls)
    except Exception:
        return None


def remove_shell_commands(code_action: str) -> Tuple[str, str]:
    """
    Remove shell commands from code and return them separately.
    """
    shell_cmds = []
    filtered_lines = []

    for line in code_action.split("\n"):
        if line.startswith("!"):
            shell_cmds.append(line)
        else:
            filtered_lines.append(line)

    return "\n".join(shell_cmds), "\n".join(filtered_lines)


def function_to_json(func) -> dict:
    """
    Converts a Python function into a JSON-serializable dictionary
    that describes the function's signature, including its name,
    description, and parameters.

    Args:
        func: The function to be converted.

    Returns:
        A dictionary representing the function's signature in JSON format.
    """
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
        type(None): "null",
    }

    try:
        signature = inspect.signature(func)
    except ValueError as e:
        raise ValueError(
            f"Failed to get signature for function {func.__name__}: {str(e)}"
        )

    parameters = {}
    for param in signature.parameters.values():
        try:
            param_type = type_map.get(param.annotation, "string")
        except KeyError as e:
            raise KeyError(
                f"Unknown type annotation {param.annotation} for parameter {param.name}: {str(e)}"
            )
        parameters[param.name] = {"type": param_type}

    required = [
        param.name
        for param in signature.parameters.values()
        if param.default == inspect._empty
    ]

    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": func.__doc__ or "",
            "parameters": {
                "type": "object",
                "properties": parameters,
                "required": required,
            },
        },
    }


# Deserialize JSON to function
def json_to_function(func_json: dict, implementation: Callable) -> Callable:
    """
    Reconstructs a function from JSON and links it to a provided implementation.

    Args:
        func_json: The JSON structure representing the function's signature.
        implementation: A Python function to use as the actual implementation.

    Returns:
        A callable Python function reconstructed from the JSON.
    """
    # Validate and parse the JSON structure using Pydantic
    func_model = FunctionModel(**func_json)
    func_name = func_model.function.name
    func_params = func_model.function.parameters.properties
    required_params = set(func_model.function.parameters.required)

    # Create parameters for the function signature
    parameters = []
    for param_name, param_data in func_params.items():
        # Map JSON types to Python types
        param_type = {
            "string": str,
            "integer": int,
            "number": float,
            "boolean": bool,
            "array": list,
            "object": dict,
            "null": type(None),
        }.get(param_data.type, Any)

        default = inspect.Parameter.empty if param_name in required_params else None
        parameters.append(
            inspect.Parameter(
                name=param_name,
                kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                annotation=param_type,
                default=default,
            )
        )

    # Create the function signature dynamically
    signature = inspect.Signature(parameters)

    # Define a function with the dynamic signature
    def dynamic_func(*args, **kwargs):
        """Dynamically created function"""
        bound_args = signature.bind(*args, **kwargs)
        bound_args.apply_defaults()
        # Call the provided implementation with bound arguments
        return implementation(*bound_args.args, **bound_args.kwargs)

    # Attach attributes to mimic the original function
    dynamic_func.__name__ = func_name
    dynamic_func.__doc__ = func_model.function.description
    dynamic_func.__signature__ = signature  # type: ignore[attr-defined]

    return dynamic_func


def openapi_to_functions(openapi_spec):
    functions = []

    for path, methods in openapi_spec["paths"].items():
        for method, spec_with_ref in methods.items():
            # 1. Resolve JSON references.
            spec = jsonref.replace_refs(spec_with_ref)

            # 2. Extract a name for the functions.
            function_name = spec.get("operationId")

            # 3. Extract a description and parameters.
            desc = spec.get("description") or spec.get("summary", "")

            schema = {"type": "object", "properties": {}}

            req_body = (
                spec.get("requestBody", {})
                .get("content", {})
                .get("application/json", {})
                .get("schema")
            )
            if req_body:
                schema["properties"]["requestBody"] = req_body

            params = spec.get("parameters", [])
            if params:
                param_properties = {
                    param["name"]: param["schema"]
                    for param in params
                    if "schema" in param
                }
                schema["properties"]["parameters"] = {
                    "type": "object",
                    "properties": param_properties,
                }

            functions.append(
                {
                    "type": "function",
                    "function": {
                        "name": function_name,
                        "description": desc,
                        "parameters": schema,
                    },
                }
            )

    return functions


def collision_check_decorator(func):
    """
    Decorator that checks for variable name collisions with global functions or imported modules.
    """

    @wraps(func)
    def wrapper(code, global_vars):
        # Parse the code to get the assignments
        tree = ast.parse(code)

        # Collect all function names and imported module names from builtins and current global namespace
        existing_names = set(dir(builtins))  # Built-in functions
        existing_names.update(
            global_vars.keys()
        )  # Global variables/functions in the namespace

        # Check for imports and function definitions in the code
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                existing_names.add(node.name)  # Function names
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    existing_names.add(alias.name)  # Imported module name
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    existing_names.add(alias.name)  # Imported function name or module

        # Parse the code to find the assigned variables
        assigned_vars = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = (
                    node.targets if isinstance(node, ast.Assign) else [node.target]
                )
                for target in targets:
                    if isinstance(target, ast.Name):
                        assigned_vars.add(target.id)
            elif isinstance(node, ast.AugAssign):
                if isinstance(node.target, ast.Name):
                    assigned_vars.add(node.target.id)

        # Check for collisions between assigned variables and existing names
        for var in assigned_vars:
            if var in existing_names:
                raise ValueError(
                    f"Collision detected: '{var}' is already in use as a function or import."
                )

        # If no collisions, execute the code
        return func(code, global_vars)

    return wrapper


@collision_check_decorator
def execute_with_collision_check(code, global_vars):
    exec(code, global_vars)


def deterministic_uuid(content: Union[str, bytes]) -> str:
    """Creates deterministic UUID on hash value of string or byte content.

    Args:
        content: String or byte representation of data.

    Returns:
        UUID of the content.
    """
    if isinstance(content, str):
        content_bytes = content.encode("utf-8")
    elif isinstance(content, bytes):
        content_bytes = content
    else:
        raise ValueError(f"Content type {type(content)} not supported !")

    hash_object = hashlib.sha256(content_bytes)
    hash_hex = hash_object.hexdigest()
    namespace = uuid.UUID("00000000-0000-0000-0000-000000000000")
    content_uuid = str(uuid.uuid5(namespace, hash_hex))

    return content_uuid


def is_sql_valid(sql: str) -> bool:
    """
    Checks if the SQL query is valid. This is usually used to check if we should run the SQL query or not.
    By default it checks if the SQL query is a SELECT statement. You can override this method to enable running other types of SQL queries.

    Args:
        sql (str): The SQL query to check.

    Returns:
        bool: True if the SQL query is valid, False otherwise.
    """

    parsed = sqlparse.parse(sql)

    for statement in parsed:
        if statement.get_type() == "SELECT":
            return True

    return False


def load_dependencies(dependencies, global_env):
    """
    Load dependencies into the global_env based on the provided import statements.
    """
    try:
        parsed = ast.parse(dependencies)
        for node in parsed.body:
            if isinstance(node, ast.Import):  # Handle import statements
                for alias in node.names:
                    module = importlib.import_module(alias.name)
                    asname = alias.asname or alias.name
                    global_env[asname] = module
            elif isinstance(node, ast.ImportFrom):  # Handle from ... import ...
                module = importlib.import_module(node.module)
                for alias in node.names:
                    obj_name = alias.name
                    asname = alias.asname or obj_name
                    global_env[asname] = getattr(module, obj_name)
        print("Dependencies loaded successfully.")
    except Exception as e:
        InterpreterEnvUpdateError(f"Error loading dependencies:\n{dependencies}\n{e}")


def process_function_code(code, global_env):
    """
    Compile and execute the provided function code in the given global_env.
    """
    try:
        exec(compile(code, filename="<string>", mode="exec"), global_env)
        print("Function code processed successfully.")
    except Exception as e:
        print(f"Error processing function code:\n{code}\n{e}")
