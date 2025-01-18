import json
import os.path
import pickle
import base64
from typing import List

import chromadb
import pandas as pd
from chromadb.config import Settings
from chromadb.utils import embedding_functions

from ..agent import AbstractPythonCodeAgent
from ..utils import deterministic_uuid
from ..resources.prompts import format_action_jinja
from ..types import GeneratedTool
from ..exceptions import ToolLoadingFromDiskError
import glob

default_ef = embedding_functions.DefaultEmbeddingFunction()


class ChromaDB_VectorStore(AbstractPythonCodeAgent):
    def __init__(self, config=None):
        AbstractPythonCodeAgent.__init__(self, config=config)
        if config is None:
            config = {}

        manual_scripts_path = config.get("manual_scripts_path", None)
        path = config.get("path", ".")
        self.embedding_function = config.get("embedding_function", default_ef)
        curr_client = config.get("client", "persistent")
        collection_metadata = config.get("collection_metadata", None)
        self.n_results_tools = config.get("n_results_tools", config.get("n_results", 2))
        self.n_results_sql = config.get("n_results_sql", config.get("n_results", 2))
        self.n_results_documentation = config.get(
            "n_results_documentation", config.get("n_results", 2)
        )
        self.n_results_ddl = config.get("n_results_ddl", config.get("n_results", 2))

        if curr_client == "persistent":
            self.chroma_client = chromadb.PersistentClient(
                path=path, settings=Settings(anonymized_telemetry=False)
            )
        elif curr_client == "in-memory":
            self.chroma_client = chromadb.EphemeralClient(
                settings=Settings(anonymized_telemetry=False)
            )
        elif isinstance(curr_client, chromadb.api.client.Client):
            # allow providing client directly
            self.chroma_client = curr_client
        else:
            raise ValueError(f"Unsupported client was set in config: {curr_client}")

        self.documentation_collection = self.chroma_client.get_or_create_collection(
            name="documentation",
            embedding_function=self.embedding_function,
            metadata=collection_metadata,
        )
        self.tool_collection = self.chroma_client.get_or_create_collection(
            name="tool",
            embedding_function=self.embedding_function,
            metadata=collection_metadata,
        )
        self.sql_collection = self.chroma_client.get_or_create_collection(
            name="sql",
            embedding_function=self.embedding_function,
            metadata=collection_metadata,
        )
        self.ddl_collection = self.chroma_client.get_or_create_collection(
            name="ddl",
            embedding_function=self.embedding_function,
            metadata=collection_metadata,
        )
        if manual_scripts_path:
            self.add_tools_from_path(manual_scripts_path)

    def generate_embedding(self, data: str, **kwargs) -> List[float]:
        embedding = self.embedding_function([data])
        if len(embedding) == 1:
            return embedding[0]
        return embedding

    @AbstractPythonCodeAgent.auto_update_global_env
    def add_tool(self, tool: GeneratedTool, **kwargs) -> str:
        id = deterministic_uuid(tool.name) + "-tool"
        res = self.tool_collection.get(ids=[id])
        if res["ids"]:
            self.tool_collection.delete(ids=[tool.name])
        self.tool_collection.add(
            # serialize
            documents=base64.b64encode(pickle.dumps(tool)).decode("utf-8"),
            embeddings=self.generate_embedding(tool.description),
            ids=id,
        )
        return id

    def add_question_sql(self, question: str, sql: str, **kwargs) -> str:
        question_sql_json = json.dumps(
            {
                "question": question,
                "sql": sql,
            },
            ensure_ascii=False,
        )
        id = deterministic_uuid(question_sql_json) + "-sql"
        self.sql_collection.add(
            documents=question_sql_json,
            embeddings=self.generate_embedding(question_sql_json),
            ids=id,
        )

        return id

    def add_ddl(self, ddl: str, **kwargs) -> str:
        id = deterministic_uuid(ddl) + "-ddl"
        self.ddl_collection.add(
            documents=ddl,
            embeddings=self.generate_embedding(ddl),
            ids=id,
        )
        return id

    def add_documentation(self, documentation: str, **kwargs) -> str:
        id = deterministic_uuid(documentation) + "-doc"
        self.documentation_collection.add(
            documents=documentation,
            embeddings=self.generate_embedding(documentation),
            ids=id,
        )
        return id

    # TODO: Maintain consistency between Vector DB collection, generated_tools and interpreter global env
    def remove_collection(self, collection_name: str) -> bool:
        """
        This function can reset the collection to empty state.

        Args:
            collection_name (str): sql or ddl or documentation

        Returns:
            bool: True if collection is deleted, False otherwise
        """
        if collection_name == "tool":
            self.chroma_client.delete_collection(name="tool")
            self.tool_collection = self.chroma_client.get_or_create_collection(
                name="tool", embedding_function=self.embedding_function
            )
            return True
        if collection_name == "sql":
            self.chroma_client.delete_collection(name="sql")
            self.sql_collection = self.chroma_client.get_or_create_collection(
                name="sql", embedding_function=self.embedding_function
            )
            return True
        elif collection_name == "ddl":
            self.chroma_client.delete_collection(name="ddl")
            self.ddl_collection = self.chroma_client.get_or_create_collection(
                name="ddl", embedding_function=self.embedding_function
            )
            return True
        elif collection_name == "documentation":
            self.chroma_client.delete_collection(name="documentation")
            self.documentation_collection = self.chroma_client.get_or_create_collection(
                name="documentation", embedding_function=self.embedding_function
            )
            return True
        else:
            return False

    @staticmethod
    def _extract_documents(query_results) -> list:
        """
        Static method to extract the documents from the results of a query.

        Args:
            query_results (pd.DataFrame): The dataframe to use.

        Returns:
            List[str] or None: The extracted documents, or an empty list or
            single document if an error occurred.
        """
        if query_results is None:
            return []

        if "documents" in query_results:
            documents = query_results["documents"]

            if len(documents) == 1 and isinstance(documents[0], list):
                try:
                    documents = [json.loads(doc) for doc in documents[0]]
                except Exception:
                    return documents[0]

            return documents

    def get_similar_question_sql(self, question: str, **kwargs) -> list:
        return ChromaDB_VectorStore._extract_documents(
            self.sql_collection.query(
                query_texts=[question],
                n_results=self.n_results_sql,
            )
        )

    def get_tools(self, query: str, **kwargs) -> List[GeneratedTool]:
        tools = ChromaDB_VectorStore._extract_documents(
            self.tool_collection.query(
                query_texts=[query],
                n_results=self.n_results_tools,
            )
        )
        # deserialize
        tools = [pickle.loads(base64.b64decode(tool)) for tool in tools]
        return tools

    def get_relevant_tools(self, query: str) -> str:
        relevant_tools: List[GeneratedTool] = self.get_tools(query)
        if relevant_tools:
            return "\n".join([format_action_jinja(tool) for tool in relevant_tools])
        else:
            return "No tool found"

    # def get_similar_question_sql(self, question: str, **kwargs) -> list:
    #     return ChromaDB_VectorStore._extract_documents(
    #         self.sql_collection.query(
    #             query_texts=[question],
    #             n_results=self.n_results_sql,
    #         )
    #     )

    def get_related_ddl(self, question: str, **kwargs) -> list:
        return ChromaDB_VectorStore._extract_documents(
            self.ddl_collection.query(
                query_texts=[question],
                n_results=self.n_results_ddl,
            )
        )

    def get_related_documentation(self, question: str, **kwargs) -> list:
        return ChromaDB_VectorStore._extract_documents(
            self.documentation_collection.query(
                query_texts=[question],
                n_results=self.n_results_documentation,
            )
        )

    def add_tools_from_path(self, path):
        files_paths = glob.glob(os.path.join(path, "*.py"))
        for file_path in files_paths:
            try:
                with open(file_path, "r") as fp_:
                    code = fp_.read()
                tools = self.parse_functions_from_code(code)
                self.add_tools(tools)
            except Exception as e:
                raise ToolLoadingFromDiskError(
                    "Check the path to the python scripts and their formats."
                ) from e
