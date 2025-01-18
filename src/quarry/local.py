import math
from pprint import pprint

from quarry.hf_agent import HfAgent
from quarry.openai_agent import OpenAIAgent
from quarry.anthropic import AnthropicAgent
from quarry.ollama_agent import OllamaAgent
from quarry.chroma_db import ChromaDB_VectorStore
from quarry.resources.prompts import (
    extract_python_codes,
)
import pandas as pd

from quarry.search import (
    ReActWorldModel,
    ReActConfig,
    Reasoner,
)
from quarry.algorithm import MCTS

import pickle
import base64
import re


class LocalHfAgent(ChromaDB_VectorStore, HfAgent):
    def __init__(self, config=None):
        ChromaDB_VectorStore.__init__(self, config=config)
        HfAgent.__init__(self, config=config)


class LocalOpenAIAgent(ChromaDB_VectorStore, OpenAIAgent):
    def __init__(self, config=None):
        ChromaDB_VectorStore.__init__(self, config=config)
        OpenAIAgent.__init__(self, config=config)


class LocalAnthropicAgent(ChromaDB_VectorStore, AnthropicAgent):
    def __init__(self, config=None):
        ChromaDB_VectorStore.__init__(self, config=config)
        AnthropicAgent.__init__(self, config=config)


class LocalOllamaAgent(ChromaDB_VectorStore, OllamaAgent):
    def __init__(self, config=None):
        ChromaDB_VectorStore.__init__(self, config=config)
        OllamaAgent.__init__(self, config=config)


def extract_code_blocks(text):
    # Regex pattern to match text between triple backticks
    pattern = r"```(.*?)```"
    # Extract all matches with DOTALL to include newlines
    return re.findall(pattern, text, re.DOTALL)


def demo():
    import os

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    try:
        print("Trying to load .env")
        from dotenv import load_dotenv

        load_dotenv()
    except Exception as e:
        print(f"Failed to load .env {e}")
        pass

    # agent = LocalHfAgent(
    #     {
    #         "model_name_or_path": "google/gemma-2-9b-it",
    #         "device": "mps",
    #         "path": "chroma_dir",
    #         "n_results": 10,
    #     }
    # )
    # agent = LocalOpenAIAgent(
    #     {
    #         "model_name_or_path": "gpt-4o-mini",
    #         "path": "chroma_dir",
    #         "n_results_documentation": 20,
    #         "n_results_tools": 5,
    #         "openai_api_key": os.getenv("OPENAI_API_KEY"),
    #     }
    # )

    agent = LocalAnthropicAgent(
        {
            "model_name_or_path": "claude-3-5-sonnet-20241022",
            "path": "chroma_dir",
            "n_results_documentation": 20,
            "n_results_tools": 5,
            "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY"),
        }
    )

    # agent = LocalOllamaAgent(
    #     {
    #         "model": "phi4", #options:qwq:gemma2:9b-instruct-q4_0:phi4
    #         "path": "chroma_dir",
    #         "n_results_documentation": 5,
    #         "n_results_tools": 5,
    #     }
    # )

    # agent.remove_collection("tool")  # if need to remove tool collection

    agent.connect_to_postgres(
        host="localhost",  # os.getenv("DB_HOST"),
        dbname=os.getenv("DB_USER"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        port=os.getenv("DB_PORT"),
    )

    def save_final_answer_as_csv(df: pd.DataFrame, file_name: str) -> None:
        print("Finished")
        df.to_csv(f"workdir/{file_name}", index=False)

    func_map = {
        "get_relevant_tools": agent.get_relevant_tools,
        "save_final_answer_as_csv": save_final_answer_as_csv,
    }
    agent.register_tools(
        func_map
    )  # adding codes directly from python env. Also good for adding agents and more complex tools.

    agent.add_tools_from_path(
        "src/quarry/resources/scripts"
    )  # for simpler on disk functions.
    # Load all the existing tools in python's interpreter global env
    tools = agent.tool_collection.get()["documents"]
    tools = [pickle.loads(base64.b64decode(tool)) for tool in tools]
    for tool in tools:
        pprint(tool)
        agent.update_global_env(tool)  # Loading saved tool:GeneratedTool in VectorDB.

    goal = "Design an advanced security scanner engine for my mac"
    subgoal_prompt = """
Goal Decomposition - Data Exploration to Analysis
----------------------------------------------
Original Goal: {goal}

Analysis Approach:
1. Start with data discovery and understanding
2. Progress through increasingly complex analysis
3. Conclude with goal-specific insights

Format Guidelines:
1. Each subgoal must:
   - Start with a clear action verb
   - Maximum 10 words
   - Progress logically from data understanding to insights
2. One subgoal per line with no number

Progression Steps Must Include:
1. Identify relevant tables and their relationships
2. Examine table schemas and data structures
3. Build up analysis complexity gradually
4. Reach final analytical goal


Example:
goal: Analyze user access patterns to identify potential insider threats
subgoals:
Identify tables containing user activity and access logs
Examine schema for user, access, and permission tables
Map relationships between user and activity tables
Extract basic user access patterns and frequencies
Calculate baseline activity metrics per user role
Identify deviations from normal access patterns
Correlate unusual activities across different tables
Flag high-risk behavior patterns by severity
Generate comprehensive insider threat analysis report

Please list your sequential subgoals below:
"""
    subgoals = agent.submit_prompt(
        [agent.user_message(subgoal_prompt.format(goal=goal))], max_tokens = 128)
    print(subgoals)
    subgoals = subgoals.split("\n")
    subgoals = [subgoal for subgoal in subgoals if subgoal]

    for subgoal in subgoals:
        print(subgoal)
        world_model = ReActWorldModel(agent=agent, goal=subgoal, max_iterations=7)
        config = ReActConfig(
            agent=agent, num_actions=3, max_new_tokens=512, temperature=0.3
        )

        algorithm = MCTS(
            depth_limit=7,
            w_exp=math.sqrt(2),
            disable_tqdm=False,
            output_trace_in_each_iter=False,
            n_iters=3,
            goal=subgoal,
        )
        reasoner_rap = Reasoner(
            world_model=world_model, search_config=config, search_algo=algorithm
        )
        result_rap = reasoner_rap(subgoal)
        react_result = result_rap.terminal_state.action

        from colorama import init, Fore, Style

        init()

        def highlight_interpreter_trace(text):
            pattern = r"(Interpreter trace:.*?)(?=(\n\s*['\"]?\s*thought\s*['\"]?:))"
            highlighted_text = re.sub(
                pattern,
                lambda m: f"{Fore.GREEN}{m.group(1)}{Style.RESET_ALL}",
                text,
                flags=re.IGNORECASE | re.DOTALL,
            )
            return highlighted_text

        print(highlight_interpreter_trace(react_result))

        # if only add at the end.
        code_blocks = react_result.split("```python")
        code_blocks = ["```python" + code_block for code_block in code_blocks][1:]
        for code_block in code_blocks:
            # Add tools if successful output
            try:
                if "Error:" not in code_block and "Empty DataFrame" not in code_block:
                    code = extract_python_codes(code_block)[0]
                    tools = agent.parse_functions_from_code(code)
                    agent.add_tools(tools)
            except Exception as e:
                print(e)
                continue

        # Max : num_actions* depth_limit*n_iters* 2(one for generation one for eval)
        print(f"LLM model was called {agent.counter.count} times")
