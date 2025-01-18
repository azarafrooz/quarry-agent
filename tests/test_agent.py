import os
import pytest

from quarry.hf_agent import HfAgent
from quarry.ollama_agent import OllamaAgent
from quarry.openai_agent import openai_agent
from quarry.anthropic import AnthropicAgent
from quarry.chroma_db import ChromaDB_VectorStore
from quarry.utils import process_function_code, load_dependencies

import importlib
from quarry.resources.prompts import (
    format_action,
    format_action_jinja,
    extract_python_codes,
)

try:
    print("Trying to load .env")
    from dotenv import load_dotenv

    load_dotenv()
except Exception as e:
    print(f"Failed to load .env {e}")
    pass

# @pytest.fixture()
# def setup_agent():
#     class LocalHfAgent(ChromaDB_VectorStore, HfAgent):
#         def __init__(self, config=None):
#             ChromaDB_VectorStore.__init__(self, config=config)
#             HfAgent.__init__(self, config=config)
#
#     local_hf_agent = LocalHfAgent(
#         {"model_name_or_path": "google/gemma-2-9b-it", "device": "mps"}
#     )
#     local_hf_agent.connect_to_sqlite(os.getenv("FILE_DATABASE"))
#     return local_hf_agent


@pytest.fixture()
def setup_agent():
    class LocalOllamaAgent(ChromaDB_VectorStore, OllamaAgent):
        def __init__(self, config=None):
            ChromaDB_VectorStore.__init__(self, config=config)
            OllamaAgent.__init__(self, config=config)

    local_ollama_agent = LocalOllamaAgent(
        {
            "model": "gemma2:9b-instruct-q4_0",
            "path": "tests/chroma_dir",
            "n_results_tools": 10,
        }
    )
    local_ollama_agent.connect_to_postgres(
        host=os.getenv("DB_HOST"),
        dbname=os.getenv("DB_USER"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        port=os.getenv("DB_PORT"),
    )
    return local_ollama_agent


def test_get_schema_info(setup_agent):
    code = '''
from typing import Union
import pandas as pd
import sqlite3
def run_sql_sqlite(sql: str, conn: sqlite3.Connection) -> Union[pd.DataFrame, None]:
    """Executes a SQLite Query and returns results as a DataFrame or None when facing errors."""
    return pd.read_sql_query(sql, conn)
run_sql_sqlite("SELECT * from events where event_type='AwsApiCall'", conn)
run_sql_sqlite("SELECT name AS table_name, sql AS schema FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';", conn)
'''
    assert setup_agent.execute_code(code) != ""


def test_embedding(setup_agent):
    assert len(setup_agent.generate_embedding("hello world")) > 1


@pytest.mark.skip("For faster testing")
def test_llm_response(setup_agent):
    message = """
    write a python function for the following SQL using f-string so it is generic and can be utilized later
    SELECT * FROM events Where IP=101.10.0.1
    """
    prompt = [setup_agent.user_message(message)]
    assert (
        setup_agent.submit_prompt(
            prompt, max_new_tokens=32, temperature=0.2, do_sample=True
        )
        == ""
    )


@pytest.fixture(params=["tests/expected_output/interpreter_1.output"])
def expected_output_1(request):
    file_path = request.param
    with open(file_path, "r") as fp_:
        return fp_.read()


def test_python_interpreter_output_1(expected_output_1, setup_agent):
    # Execute code and capture output
    captured_output = setup_agent.execute_code(
        """
x = 10
y = 20
x + y
if x > 5:
    print(f"x is greater than 5: {x}")
try:
    1 / 0
except ZeroDivisionError:
    print("Caught division by zero!")

def pow(z):
    return z**2

for i in range(5):
    if i == 3:
        continue
    print(pow(i))
"""
    )

    # Assert the output matches the expected output
    assert (
        expected_output_1.strip() == captured_output.strip()
    ), f"Captured output does not match expected. \nCaptured: {captured_output}\nExpected: {expected_output_1}"


@pytest.fixture(params=["tests/expected_output/interpreter_2.output"])
def expected_output_2(request):
    file_path = request.param
    with open(file_path, "r") as fp_:
        return fp_.read()


def test_python_interpreter_output_2(expected_output_2, setup_agent):
    # Execute code and capture output
    captured_output = setup_agent.execute_code(
        """
from rank_bm25 import BM25Okapi
corpus = [
   "Hello there good man!",
   "It is quite windy in London",
   "How is the weather today?"
]
tokenized_corpus = []
for doc in corpus:
   doc_tokens = doc.split()
   tokenized_corpus.append(doc_tokens)
print(tokenized_corpus)
bm25 = BM25Okapi(tokenized_corpus)
query = "windy London"
tokenized_query = query.split(" ")
bm25.get_top_n(tokenized_query, corpus, n=1)
"""
    )

    # Assert the output matches the expected output
    assert (
        expected_output_2.strip() == captured_output.strip()
    ), f"Captured output does not match expected. \nCaptured: {captured_output}\nExpected: {expected_output_2}"


def test_python_interpreter_collision(setup_agent):
    """
    Test the collision with func name, yet to figure out the import as ...
    """

    def func():
        return 42

    setup_agent.global_env["func"] = func()

    code = """
func = 42  # This will collide with defined function name
"""
    captured_output = setup_agent.execute_code(code)
    assert (
        "Collision detected: 'func' is already in use as a function or import."
        in captured_output
    )


# @pytest.mark.skip()
def test_python_interpreter_import_collision(setup_agent):
    """
    Test the collision with func name, yet to figure out the import as ...
    """

    setup_agent.global_env["pd"] = importlib.import_module("pandas")

    code = """
pd = 42  # This will collide with the imported 'pandas' module.
"""
    captured_output = setup_agent.execute_code(code)
    assert (
        "Collision detected: 'pd' is already in use as a function or import."
        in captured_output
    )


def test_load_collision(setup_agent):
    parsed_dict = {
        "code": """
def my_function(x):
    return pd.Series([i * x for i in range(5)])
    """,
        "dependencies": "import pandas as pd\nimport numpy as np",
    }

    # Load dependencies and execute the function code
    load_dependencies(parsed_dict["dependencies"], setup_agent.global_env)
    process_function_code(parsed_dict["code"], setup_agent.global_env)
    # Access and use the dynamically loaded function
    code = """
my_function(2)
np = 2
"""
    captured_output = setup_agent.execute_code(code)
    assert "my_function" in setup_agent.global_env
    assert (
        "Collision detected: 'np' is already in use as a function or import."
        in captured_output
    )
    assert (
        """
Out [ ]:
0    0
1    2
2    4
3    6
4    8
dtype: int64
""".strip()
        in captured_output
    )


def test_analyze_code_with_ast(setup_agent):
    code = '''
import pandas as pd
from numpy import float32
def test_generate_code(query: str, ip: int)->pd.Dataframe:
    """Test code"""
    goal = "Find suspicious IAM activity"
    query = "SELECT * FROM actions WHERE type = 'IAM' AND suspicious = 1;"
    result = pd.run_sql(query)
    code = agent.generate_code(goal, query)
    assert query not in code
    return result
test_generate_code()
'''
    tools = setup_agent.parse_functions_from_code(code)
    assert len(tools) == 1
    assert tools[0].name == "test_generate_code"
    assert tools[0].dependencies == "import pandas as pd\nfrom numpy import float32"
    assert tools[0].inputs == "query: str, ip: int"
    assert tools[0].output_type == "pd.Dataframe"

    assert (
        format_action(tools[0]).strip()
        == "- test_generate_code(query: str, ip: int) -> pd.Dataframe: Test code"
    )
    assert (
        format_action_jinja(tools[0]).strip()
        == "- test_generate_code(query: str, ip: int) -> pd.Dataframe: Test code"
    )


def test_cleanup_interpreter(setup_agent):
    code = '''
from typing import Union
import pandas as pd
import sqlite3
def run_sql_sqlite(sql: str, conn: sqlite3.Connection) -> Union[pd.DataFrame, None]:
    """Executes a SQLite Query and returns results as a DataFrame or None when facing errors."""
    return pd.read_sql_query(sql, conn)
run_sql_sqlite("SELECT * from events where event_type='AwsApiCall'", conn)
    '''
    with setup_agent.preserve_state():
        setup_agent.execute_code(code)
    assert "Error: name 'run_sql_sqlite' is not defined" in setup_agent.execute_code(
        """run_sql_sqlite("SELECT * from events where event_type='AwsApiCall'", conn)"""
    )

    with setup_agent.preserve_state():
        setup_agent.execute_code(code)
        assert (
            "Error: name 'run_sql_sqlite' is not defined"
            not in setup_agent.execute_code(
                """run_sql_sqlite("SELECT * from events where event_type='AwsApiCall'", conn)"""
            )
        )


def test_get_tools(setup_agent):
    setup_agent.global_env["get_relevant_tools"] = setup_agent.get_relevant_tools
    setup_agent.remove_collection("tool")
    result = setup_agent.execute_code("get_relevant_tools(query='Hello world')")
    assert "No tool found" in result


@pytest.mark.skip("Will change this to Steampipe postgres")
def test_update_tool(setup_agent):
    setup_agent.global_env["get_relevant_tools"] = setup_agent.get_relevant_tools
    setup_agent.add_tools_from_path("src/quarry/resources/scripts")
    assert (
        "Error: name 'run_sql_sqlite' is not defined"
        not in setup_agent.execute_code(
            """run_sql_sqlite("SELECT * from events where event_type='AwsApiCall'", conn)"""
        )
    )
    result = setup_agent.execute_code("get_relevant_tools(query='Do a lexical search')")
    assert (
        "lexical_search(query: str, corpus: list[str], k: int) -> list[str]" in result
    )


@pytest.mark.skip("Will change this to Steampipe postgres")
def test_update_tool(setup_agent):
    setup_agent.global_env["get_relevant_tools"] = setup_agent.get_relevant_tools
    setup_agent.add_tools_from_path("src/quarry/resources/scripts")
    assert (
        "Error: name 'run_sql_sqlite' is not defined"
        not in setup_agent.execute_code(
            """run_sql_sqlite("SELECT * from events where event_type='AwsApiCall'", conn)"""
        )
    )
    result = setup_agent.execute_code("get_relevant_tools(query='Do a lexical search')")
    assert (
        "lexical_search(query: str, corpus: list[str], k: int) -> list[str]" in result
    )


def test_tool_capabilities(setup_agent):
    result = '''
Let me help analyze potential stock investments by looking at various metrics and recent performance data.

"thought": I'll create a function to analyze multiple stocks based on key metrics like price trends, market performance, and volatility. First, let me check what data is available in the finance_quote table.

"code":
```python
import pandas as pd
def analyze_stock_metrics(symbols: list[str]) -> pd.DataFrame:
    """Analyze key metrics for given stock symbols to evaluate investment potential."""
    query = """
    SELECT 
        symbol,
        short_name,
        regular_market_price,
        regular_market_change_percent,
        fifty_two_week_low,
        fifty_two_week_high,
        regular_market_volume,
        market_cap
    FROM finance_quote
    WHERE symbol = ANY(%s)
    """

    # Looking at major tech and stable stocks
    return pd.read_sql_query(query, conn, params=[symbols])

# Analyze a mix of established companies
symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'BRK-B', 'JNJ']
df = analyze_stock_metrics(symbols)
print(df)
```
Interpreter trace:
In [ ]:
def analyze_stock_metrics(symbols: list[str]) -> pd.DataFrame:
    """Analyze key metrics for given stock symbols to evaluate investment potential."""
    query = '\n    SELECT \n        symbol,\n        short_name,\n        regular_market_price,\n        regular_market_change_percent,\n        fifty_two_week_low,\n        fifty_two_week_high,\n        regular_market_volume,\n        market_cap\n    FROM finance_quote\n    WHERE symbol = ANY(%s)\n    '
    return pd.read_sql_query(query, conn, params=[symbols])
In [ ]:
symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'BRK-B', 'JNJ']
In [ ]:
df = analyze_stock_metrics(symbols)
<string>:4: UserWarning: pandas only supports SQLAlchemy connectable (engine/connection) or database string URI or sqlite3 DBAPI2 connection. Other DBAPI2 objects are not tested. Please consider using SQLAlchemy.
Error: Execution failed on sql '
    SELECT 
        symbol,
        short_name,
        regular_market_price,
        regular_market_change_percent,
        fifty_two_week_low,
        fifty_two_week_high,
        regular_market_volume,
        market_cap
    FROM finance_quote
    WHERE symbol = ANY(%s)
    ': column "market_cap" does not exist
LINE 10:         market_cap
                 ^
HINT:  Perhaps you meant to reference the column "finance_quote.market_id".
In [ ]:
print(df)
Error: name 'df' is not defined


"thought": I see there was an error with the market_cap column. Let me modify the query to use the correct columns and add some trend analysis using the daily data.

"code":
```python
def analyze_stock_potential(symbols: list[str]) -> pd.DataFrame:
    """Analyze stocks for investment potential using current metrics and recent trends."""
    query = """
    SELECT 
        symbol,
        short_name,
        regular_market_price,
        regular_market_change_percent,
        fifty_two_week_low,
        fifty_two_week_high,
        regular_market_volume
    FROM finance_quote
    WHERE symbol = ANY(%s)
    """

    current_metrics = pd.read_sql_query(query, conn, params=[symbols])

    # Calculate position in 52-week range (0-100%)
    current_metrics['52w_position'] = (
        (current_metrics['regular_market_price'] - current_metrics['fifty_two_week_low']) /
        (current_metrics['fifty_two_week_high'] - current_metrics['fifty_two_week_low']) * 100
    )

    return current_metrics.sort_values('regular_market_change_percent', ascending=False)

# Analyze a diverse set of stocks
symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'BRK-B', 'JNJ', 'NVDA', 'META']
result = analyze_stock_potential(symbols)
print("Stock Analysis Results:")
print(result)
```
Interpreter trace:
In [ ]:
def analyze_stock_potential(symbols: list[str]) -> pd.DataFrame:
    """Analyze stocks for investment potential using current metrics and recent trends."""
    query = '\n    SELECT \n        symbol,\n        short_name,\n        regular_market_price,\n        regular_market_change_percent,\n        fifty_two_week_low,\n        fifty_two_week_high,\n        regular_market_volume\n    FROM finance_quote\n    WHERE symbol = ANY(%s)\n    '
    current_metrics = pd.read_sql_query(query, conn, params=[symbols])
    current_metrics['52w_position'] = (current_metrics['regular_market_price'] - current_metrics['fifty_two_week_low']) / (current_metrics['fifty_two_week_high'] - current_metrics['fifty_two_week_low']) * 100
    return current_metrics.sort_values('regular_market_change_percent', ascending=False)
In [ ]:
symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'BRK-B', 'JNJ', 'NVDA', 'META']
Error: Collision detected: 'symbols' is already in use as a function or import.
In [ ]:
result = analyze_stock_potential(symbols)
In [ ]:
print('\nStock Analysis Results:')

Stock Analysis Results:
In [ ]:
print(result)
  symbol                   short_name  regular_market_price  regular_market_change_percent  fifty_two_week_low  fifty_two_week_high  regular_market_volume  52w_position
3   MSFT        Microsoft Corporation                424.56                       0.518503              380.38               468.35               14863983     50.221666
2   AAPL                   Apple Inc.                242.70                       0.202300              164.08               260.10               36473385     81.878775
1   AMZN             Amazon.com, Inc.                222.13                       0.009006              149.91               233.00               23698805     86.917800
4  BRK-B  Berkshire Hathaway Inc. New                451.84                      -0.238456              357.98               491.67                2663030     70.207196
5  GOOGL                Alphabet Inc.                193.95                      -0.787768              130.67               201.42               24249066     89.441696
0    JNJ            Johnson & Johnson                142.27                      -2.708060              141.44               168.85                9022564      3.028092


"thought": Based on the analysis, I can provide a recommendation. Let me create a function to evaluate and score each stock based on multiple factors.

"code":
```python
def score_stocks(df: pd.DataFrame) -> pd.DataFrame:
    """Score stocks based on multiple factors for investment recommendation."""

    # Normalize scores between 0 and 1 for each metric
    df['momentum_score'] = (df['regular_market_change_percent'] - df['regular_market_change_percent'].min()) / \
                          (df['regular_market_change_percent'].max() - df['regular_market_change_percent'].min())

    # Value score (inverse of 52w position - lower is better as it indicates more upside potential)
    df['value_score'] = 1 - (df['52w_position'] / 100)

    # Calculate final score (equal weights)
    df['final_score'] = (df['momentum_score'] + df['value_score']) / 2

    return df.sort_values('final_score', ascending=False)

scored_stocks = score_stocks(result)
print("\nFinal Stock Rankings:")
print(scored_stocks[['symbol', 'short_name', 'regular_market_price', 'regular_market_change_percent', 'final_score']])
```
Interpreter trace:
In [ ]:
def score_stocks(df: pd.DataFrame) -> pd.DataFrame:
    """Score stocks based on multiple factors for investment recommendation."""
    df['momentum_score'] = (df['regular_market_change_percent'] - df['regular_market_change_percent'].min()) / (df['regular_market_change_percent'].max() - df['regular_market_change_percent'].min())
    df['value_score'] = 1 - df['52w_position'] / 100
    df['final_score'] = (df['momentum_score'] + df['value_score']) / 2
    return df.sort_values('final_score', ascending=False)
In [ ]:
scored_stocks = score_stocks(result)
In [ ]:
print('Final Stock Rankings:')

Final Stock Rankings:
In [ ]:
print(scored_stocks[['symbol', 'short_name', 'regular_market_price', 'regular_market_change_percent', 'final_score']])
  symbol                   short_name  regular_market_price  regular_market_change_percent  final_score
3   MSFT        Microsoft Corporation                424.56                       0.518503     0.748892
2   AAPL                   Apple Inc.                242.70                       0.202300     0.541606
4  BRK-B  Berkshire Hathaway Inc. New                451.84                      -0.238456     0.531663
1   AMZN             Amazon.com, Inc.                222.13                       0.009006     0.486458
0    JNJ            Johnson & Johnson                142.27                      -2.708060     0.484860
5  GOOGL                Alphabet Inc.                193.95                      -0.787768     0.350367


"thought": Based on the comprehensive analysis, I can now provide a final recommendation.

Let me submit the final answer with a detailed explanation of the best investment options.

"code":
```python
recommendation = """
Based on the analysis of major stocks, here are the top investment recommendations:

1. Microsoft (MSFT) - Best Overall Choice
   - Strong positive momentum (0.52% recent growth)
   - Balanced position in its 52-week range (50.22%)
   - Solid trading volume indicating market interest
   - Highest combined score for momentum and value

2. Apple (AAPL) - Strong Runner-up
   - Positive momentum (0.20% growth)
   - Trading at 81.88% of its 52-week range
   - Highest trading volume indicating strong market participation

3. Berkshire Hathaway (BRK-B) - Stable Value Option
   - More conservative choice
   - Trading at 70.21% of its 52-week range
   - Lower volatility and stable performance

Key considerations:
- Microsoft shows the best balance of momentum and value
- All recommended stocks show strong fundamentals and market presence
- Consider diversifying across these options rather than investing in just one

Remember to:
- Conduct additional research on company fundamentals
- Consider your personal risk tolerance and investment timeline
- Consult with a financial advisor for personalized advice
"""

submit_final_answer(recommendation)
```
Interpreter trace:
In [ ]:
recommendation = '\nBased on the analysis of major stocks, here are the top investment recommendations:\n\n1. Microsoft (MSFT) - Best Overall Choice\n   - Strong positive momentum (0.52% recent growth)\n   - Balanced position in its 52-week range (50.22%)\n   - Solid trading volume indicating market interest\n   - Highest combined score for momentum and value\n\n2. Apple (AAPL) - Strong Runner-up\n   - Positive momentum (0.20% growth)\n   - Trading at 81.88% of its 52-week range\n   - Highest trading volume indicating strong market participation\n\n3. Berkshire Hathaway (BRK-B) - Stable Value Option\n   - More conservative choice\n   - Trading at 70.21% of its 52-week range\n   - Lower volatility and stable performance\n\nKey considerations:\n- Microsoft shows the best balance of momentum and value\n- All recommended stocks show strong fundamentals and market presence\n- Consider diversifying across these options rather than investing in just one\n\nRemember to:\n- Conduct additional research on company fundamentals\n- Consider your personal risk tolerance and investment timeline\n- Consult with a financial advisor for personalized advice\n'
In [ ]:
submit_final_answer(recommendation)
Out [ ]:
Final Answer: 
Based on the analysis of major stocks, here are the top investment recommendations:

1. Microsoft (MSFT) - Best Overall Choice
   - Strong positive momentum (0.52% recent growth)
   - Balanced position in its 52-week range (50.22%)
   - Solid trading volume indicating market interest
   - Highest combined score for momentum and value

2. Apple (AAPL) - Strong Runner-up
   - Positive momentum (0.20% growth)
   - Trading at 81.88% of its 52-week range
   - Highest trading volume indicating strong market participation

3. Berkshire Hathaway (BRK-B) - Stable Value Option
   - More conservative choice
   - Trading at 70.21% of its 52-week range
   - Lower volatility and stable performance

Key considerations:
- Microsoft shows the best balance of momentum and value
- All recommended stocks show strong fundamentals and market presence
- Consider diversifying across these options rather than investing in just one

Remember to:
- Conduct additional research on company fundamentals
- Consider your personal risk tolerance and investment timeline
- Consult with a financial advisor for personalized advice
'''
    setup_agent.remove_collection("tool")

    code_blocks = result.split("```python")
    code_blocks = ["```python" + code_block for code_block in code_blocks][1:]
    assert len(code_blocks) == 4
    for code_block in code_blocks:
        code = extract_python_codes(code_block)[0]
        tools = setup_agent.parse_functions_from_code(code)
        setup_agent.add_tools(tools)

    assert (
        """
- analyze_stock_potential(symbols: list[str]) -> pd.DataFrame: Analyze stocks for investment potential using current metrics and recent trends.
- analyze_stock_metrics(symbols: list[str]) -> pd.DataFrame: Analyze key metrics for given stock symbols to evaluate investment potential.
- score_stocks(df: pd.DataFrame) -> pd.DataFrame: Score stocks based on multiple factors for investment recommendation.""".strip()
        in setup_agent.get_relevant_tools("Finance Analysis")
    )
