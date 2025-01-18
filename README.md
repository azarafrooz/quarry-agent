# 🚀 Quarry: Simplifying LLM Agents with a SQL/Python-First Approach

**Quarry** is your all-in-one solution for building powerful agents using [steampipe](https://steampipe.io/) with a SQL/Python-First approach.
No more wrestling with APIs—just install the right plugin from the [Steampipe Hub](https://hub.steampipe.io/) and start querying.

🔎 Zero ETL → Fetch real-time data from APIs directly into SQL.

🔄 Modular & Extensible → Combine SQL pipelines and Python functions to build intelligent agents.

🧠 Smart Reasoning → Leverage Monte Carlo Tree Search (MCTS) for advanced decision-making.

## ⚙️ Key Principles
- **SQL + Python Only**: Simple, clean, and powerful—no extra tools needed.
- **Modular Tool Learning**: Build and refine Python functions as reusable tools. 
- **Advanced Reasoning**: MCTS with self-evaluation improves code generation and task-solving. 
- **Human-in-the-Loop**: Seamlessly review, correct, and enhance generated tools.

### 🌱  Optional

### Get Better Results with Curriculum Learning

Start simple. Gradually increase task complexity to improve performance and reasoning.

###  🔄 Share the previous results using CSV plugin

- Use `csv` plugin. 
- Include `def Save_results_in_CSV` tool with `workdir` being consistent with path in:

```
 ~/.steampipe/config/csv.spc
```

This way you can use the memory of previous results as another table in Postgres/Steampipe

## 🏗️ Architecture Overview

```mermaid
flowchart TD
    Start([Start]) --> InitLoad[Load Initial Python Functions\ninto Kernel Interpreter]
    InitLoad --> ConnectDB[Connect to Steampipe\nPostgres Database]
    
    subgraph TaskGeneration[Task Generation Process]
        ConnectDB --> HighGoal[Receive High Level Goal]
        HighGoal --> SubTasks[Generate Subtasks\nIncreasing Complexity]
    end
    
    SubTasks --> GetTools[Retrieve Tools\nfrom VectorDB]
    
    subgraph MCTSProcess[MCTS Process]
        GetTools --> MCTSLoop{MCTS\nIterations}
        MCTSLoop --> GenCode[Generate\nModular Code]
        GenCode --> ExecCode[Execute in\nPython Interpreter]
        ExecCode --> ReactPrompt[React Prompting]
        ReactPrompt --> MCTSLoop
        
        %% Add data persistence branch
        ExecCode --> SaveResults{Save Results?}
        SaveResults --> |Yes| SaveCSV[Save as CSV\nin Plugin Folder]
        SaveCSV --> UpdateSteampipe[Auto-Import as\nSteampipe Table]
        UpdateSteampipe --> |New Data Available| ConnectDB
        SaveResults --> |No| ReactPrompt
    end
    
    MCTSProcess --> BFSTrace[BFS over Best Trace]
    
    subgraph ToolLearning[Tool Learning & Environment Sync]
        BFSTrace --> ExtractCode[Extract Code\nfrom Best Trace]
        ExtractCode --> NewTools[Create New\nLearned Tools]
        NewTools --> |Parallel Update| SyncProcess{Synchronous\nUpdate}
        SyncProcess --> UpdateVDB[Add Tools to VectorDB]
        SyncProcess --> UpdateKernel[Add Tools to\nKernel Global Env]
    end
    
    UpdateVDB --> |Tool Ready| GetTools
    UpdateKernel --> |Environment Ready| GetTools

    %% New UI Component for Editing Tools
    subgraph UIComponent[UI/Streamlit Tool Editor]
        EditToolsUI[Streamlit UI\nfor Editing Tools]
        EditToolsUI --> FetchTools[Fetch Tools from\nVectorDB]
        FetchTools --> ModifyTools[Edit/Update/Delete Tools]
        ModifyTools --> UpdateVDB
    end

    GetTools --> EditToolsUI

    style Start fill:#f9f,stroke:#333,stroke-width:4px
    style TaskGeneration fill:#e6f3ff,stroke:#333
    style MCTSProcess fill:#fff0e6,stroke:#333
    style ToolLearning fill:#e6ffe6,stroke:#333
    style UIComponent fill:#fffbe6,stroke:#333

    classDef processNode fill:#f9f,stroke:#333,stroke-width:2px
    classDef decision fill:#FFD700,stroke:#333,stroke-width:2px
    classDef dataNode fill:#90EE90,stroke:#333,stroke-width:2px
    class MCTSLoop,SyncProcess,SaveResults decision
    class GenCode,ExecCode,ReactPrompt,ModifyTools processNode
    class SaveCSV,UpdateSteampipe,FetchTools dataNode

    %% Adding notes for clarity
    note1[Both VectorDB and Kernel\nEnvironment stay synchronized]
    note2[Results become queryable\nby future tasks]
    note3[Streamlit UI enables manual\nediting of tools in VectorDB]
    style note1 fill:#fff,stroke:#333,stroke-width:1px
    style note2 fill:#fff,stroke:#333,stroke-width:1px
    style note3 fill:#fff,stroke:#333,stroke-width:1px

    SyncProcess --> note1
    UpdateSteampipe --> note2
    EditToolsUI --> note3
```

## 🔧 Requirements

**1. Install Steampipe**

Download and install from  [steampipe](https://steampipe.io/).

**2. Install Plugins**

Install relevant plugins. For the demo (no API keys required):
```commandline
steampipe plugin install exec
steampipe plugin install finance
steampipe plugin install csv
```

**Start the service**:

`steampipe service start --show-password`

## 🚀 Running Quarry 

**Estimate LLM Cost:**

Cost ≈ num_actions * depth_limit *  n_iters * 2 LLM calls
- num_actions: Candidates per iteration 
- depth_limit: MCTS depth 
- 2: For generation + evaluation


**env variables (steampipe postgres and LLM keys)** 

Set your env variable in `.env file`:

```
DB_HOST=host.docker.internal if using docker 
DB_PORT=
DB_NAME=
DB_USER=
DB_PASSWORD=
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
```

### Run in Docker for Safety

Modify your goal prompt in src/quarry/local.py (demo code) and run:

```bash
sh quarry.sh
```


### Run the Demo without Docker

**Install UV Package Manager**

Install from [uv](https://docs.astral.sh/uv/getting-started/installation/):

```commandline
uv venv --python 3.12
uv run demo
```

## 🛠️ Inspect & Improve Learned Tools
![EditingTool](assets/editing_tools.png)

Not all generated tools will be perfect!
Use our Streamlit UI to review, edit, and enhance them:

```commandline
uv run streamlit run tool-collection-manager.py
```

## 🔄 Extend Quarry

Want to expand beyond Steampipe?
Check out [Vanna AI](https://github.com/vanna-ai/vanna) for adapting Quarry to other databases.

**🙏 Acknowledgements**
   
- [Vanna AI](https://github.com/vanna-ai/vanna)
- [LLM Reasoners](https://github.com/Ber666/llm-reasoners)
- [Adobe Dynasaur](https://github.com/adobe-research/dynasaur)
- [Self-Refine: Improving LLMs with Self-Feedback](https://arxiv.org/abs/2305.16291)

##🌱 Future releases ##

- Training codes for finetuning the base LLM models to reduce cost in MCTS search.


**⭐️ Show Your Support!**

If you find Quarry useful, please star ⭐ this repository and share it!
Feedback and contributions are always welcome. 😊
