import pytest
import jsonref
from quarry.utils import (
    parse_functions_from_code,
    openapi_to_functions,
    function_to_json,
    json_to_function,
)


# def test_analyze_code_with_ast():
#     code = '''
# import pandas as pd
# from numpy import float32
# def test_generate_code():
#     """Test code"""
#     goal = "Find suspicious IAM activity"
#     query = "SELECT * FROM actions WHERE type = 'IAM' AND suspicious = 1;"
#     pd.run_sql(query)
#     code = agent.generate_code(goal, query)
#     assert query not in code
# test_generate_code()
# '''
#     functions = parse_functions_from_code(code)
#     assert len(functions) == 1
#     assert functions[0].name == "test_generate_code"
#     assert functions[0].dependencies == "import pandas as pd\nfrom numpy import float32"


@pytest.fixture(params=["tests/openapi_samples/sample.json"])
def openapi_data(request):
    file_path = request.param
    with open(file_path, "r") as f:
        openapi_spec = jsonref.loads(
            f.read()
        )  # it's important to load with jsonref, as explained below
    return openapi_spec


def test_openapi_to_functions(openapi_data):
    funcs = openapi_to_functions(openapi_data)
    assert funcs[0].get("function").get("name") == "listEvents"


def test_function_to_json():
    result = function_to_json(test_openapi_to_functions)
    assert result["function"].get("name", "") == "test_openapi_to_functions"


def test_json_to_function():
    def example_function(a: int, b: str, c: float = 1.0) -> str:
        """This is an example function."""
        return f"Result: {a + int(c)} - {b.upper()}"

    # Convert the function to JSON
    function_json = function_to_json(example_function)
    restored_function = json_to_function(function_json, implementation=example_function)
    # Call the restored function
    result = restored_function(42, "hello", c=3.14)
    assert result == "Result: 45 - HELLO"
