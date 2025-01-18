from abc import ABC, abstractmethod
from typing import Any, List, Callable, Union, Dict
import os
import sqlite3

import pandas
import psycopg
import pandas as pd
from urllib.parse import urlparse
import requests
import ast
import contextlib
from contextlib import contextmanager
import io
import logging
from logging import getLogger
from functools import wraps
from ..types import GeneratedTool
from ..exceptions import (
    InterpreterEnvUpdateError,
    InterpreterDependencyError,
    InterpreterFunctionProcessingError,
    ToolConfigureError,
    DependencyError,
    ValidationError,
    ImproperlyConfigured,
)
from ..utils import (
    execute_with_collision_check,
    parse_functions_from_code,
    load_dependencies,
    process_function_code,
)

logger = getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)


class AbstractPythonCodeAgent(ABC):
    """
    Abstract base class for a Python code agent that:
    - Connects to databases with credentials
    - Executes code in an IPython kernel typ env and maintains global_env including connection objects and functions/tool.
    - New modular functions get added to global_env.
      - Analyzes generated code with AST to extract modular functions
    - Saves modular functions into a vector database
    - Generates embedding for various objects including functions, documents, etc.
    """

    # Types that shouldn't be saved/restored
    PRESERVED_TYPES = (sqlite3.Connection, psycopg.Connection, io.IOBase)

    def __init__(self, config=None):
        self.dialect = config.get("dialect", "SQL")
        self._conn = None
        self.global_env = {}
        self.saved_states = []

    @abstractmethod
    def generate_embedding(self, data: str, **kwargs) -> List[float]:
        pass

    @abstractmethod
    def get_tools(self, docstring: str, **kwargs) -> List[GeneratedTool]:
        """
        Retrieve relevant modular functions from the vector database based on a query.

        Args:
            docstring: Search query for the vector database.

        Returns:
            List of relevant functions.
        """
        pass

    @abstractmethod
    def get_similar_question_sql(self, question: str, **kwargs) -> list:
        """
        This method is used to get similar questions and their corresponding SQL statements.

        Args:
            question (str): The question to get similar questions and their corresponding SQL statements for.

        Returns:
            list: A list of similar questions and their corresponding SQL statements.
        """
        pass

    @abstractmethod
    def get_related_ddl(self, question: str, **kwargs) -> list:
        """
        This method is used to get related DDL statements to a question.

        Args:
            question (str): The question to get related DDL statements for.

        Returns:
            list: A list of related DDL statements.
        """
        pass

    @abstractmethod
    def get_related_documentation(self, question: str, **kwargs) -> list:
        """
        This method is used to get related documentation to a question.

        Args:
            question (str): The question to get related documentation for.

        Returns:
            list: A list of related documentation.
        """
        pass

    @abstractmethod
    def add_ddl(self, ddl: str, **kwargs) -> str:
        """
        This method is used to add a DDL statement to the training data.

        Args:
            ddl (str): The DDL statement to add.

        Returns:
            str: The ID of the training data that was added.
        """
        pass

    @abstractmethod
    def add_documentation(self, documentation: str, **kwargs) -> str:
        """
        This method is used to add documentation to the training data.

        Args:
            documentation (str): The documentation to add.

        Returns:
            str: The ID of the training data that was added.
        """
        pass

    @abstractmethod
    def add_tool(self, tool, **kwargs) -> None:
        """
        This method adds a tool to vectordb.
        Args
            tool (GeneratedTool): tool to add
        """
        pass

    def add_tools(self, tools: List[GeneratedTool]):
        return [self.add_tool(tool=tool) for tool in tools]

    @abstractmethod
    def add_question_sql(self, question: str, sql: str, **kwargs) -> str:
        """
        This method is used to add a question and its corresponding SQL query to the training data.

        Args:
            question (str): The question to add.
            sql (str): The SQL query to add.

        Returns:
            str: The ID of the training data that was added.
        """
        pass

    @abstractmethod
    def system_message(self, message: str) -> Any:
        pass

    @abstractmethod
    def user_message(self, message: str) -> Any:
        pass

    @abstractmethod
    def assistant_message(self, message: str) -> Any:
        pass

    def str_to_approx_token_count(self, string: str) -> int:
        return int(len(string) / 4)

    def register_tools(self, func_map: Dict[str, Callable[..., Any]]) -> None:
        """
        Registers tools in the agent's global environment directly.

        Args:
            func_map (Dict[str, Callable[..., Any]]): A dictionary of tool names and their corresponding functions.

        Raises:
            KeyError: If a tool with the same name already exists.
        """
        duplicates = set(func_map) & set(self.global_env)
        if duplicates:
            raise KeyError(f"Duplicate tools detected: {', '.join(duplicates)}")

        self.global_env.update(func_map)

    def parse_functions_from_code(self, code: str) -> List[GeneratedTool] | None:
        """
        Parses the code to extract GeneratedTool and raises errors
        """
        tools = parse_functions_from_code(code)

        # First check if it's a list
        if not isinstance(tools, list):
            raise ToolConfigureError(
                f"Expected list instance, got {type(tools).__name__}"
            )

        # Then check each item in the list
        for tool in tools:
            if not isinstance(tool, GeneratedTool):
                raise ToolConfigureError(
                    f"Expected GeneratedTool instance, got {type(tool).__name__}"
                )

        return tools

    def update_global_env(self, tool: GeneratedTool) -> None:
        """
        Updates the interpreter global environment with new tool.

        Args:
            tool (GeneratedTool): Function to be added to global env

        Raises:
            InterpreterEnvUpdateError: If there's an error updating the environment
            InterpreterDependencyError: If there's an error loading dependencies
            InterpreterFunctionProcessingError: If there's an error processing function code
        """
        # Load dependencies
        try:
            load_dependencies(tool.dependencies, self.global_env)
        except Exception as e:
            raise InterpreterDependencyError(
                f"Failed to load dependencies for {tool.name}"
            ) from e

        # Process function code
        try:
            process_function_code(tool.code, self.global_env)
        except Exception as e:
            raise InterpreterFunctionProcessingError(
                f"Failed to process function code for {tool.name}"
            ) from e

        logger.info(f"Successfully updated tool: {tool.name}")

    @contextmanager
    def _safe_env_update(self, tool: GeneratedTool, rollback_logic: Callable):
        """Context manager for safely updating interpreter global python environment"""
        try:
            self.update_global_env(tool)
            yield
        except Exception as e:
            rollback_logic()
            raise InterpreterEnvUpdateError("Failed to update tool") from e

    @staticmethod
    def auto_update_global_env(func):
        """
        Decorator to ensure that interpreter global_env is consistent with vectordb.

        Returns:
            The wrapped function result or raises appropriate exception
        """

        @wraps(func)
        def wrapper(self, *args, **kwargs):
            tool = kwargs.get("tool")
            if tool is None:
                raise ToolConfigureError("Tool parameter is required")
            if not isinstance(tool, GeneratedTool):
                raise ToolConfigureError(
                    f"Expected GeneratedTool instance, got {type(tool).__name__}"
                )
            try:
                # TODO: in future cases where dependencies are not frozen, logic would be mre complicated.
                old_env = self.global_env.pop(tool.name, None)

                def rollback_logic():
                    """
                    If anything goes wrong, we roll back to previous states.
                    """
                    if old_env:
                        self.global_env[tool.name] = old_env

                with self._safe_env_update(tool=tool, rollback_logic=rollback_logic):
                    # The function only runs within the managed execution block
                    return func(self, *args, **kwargs)
            except Exception as e:
                logger.error(f"Error in auto_update_global_env: {str(e)}")
                raise InterpreterEnvUpdateError(
                    "Failed to auto-update global environment"
                ) from e

        return wrapper

    def connect_to_sqlite(
        self, url: str, check_same_thread: bool = False, **kwargs: Any
    ) -> None:
        """
        Connect to a SQLite database.

        Args:
            url (str): The URL or local path of the database to connect to.
            check_same_thread (bool): Allow the connection to be accessed in multiple threads.
        Raises:
            ValueError: If the URL is invalid or unreachable.
            requests.RequestException: If the database download fails.
        """
        # Determine the database filename
        parsed_url = urlparse(url)
        db_filename = os.path.basename(parsed_url.path)

        # If URL is a web address, download the database
        if parsed_url.scheme in {"http", "https"}:
            if not os.path.exists(db_filename):
                response = requests.get(url)
                response.raise_for_status()
                with open(db_filename, "wb") as f:
                    f.write(response.content)
            url = db_filename

        # Connect to the database
        try:
            self._conn = sqlite3.connect(
                url, check_same_thread=check_same_thread, **kwargs
            )
        except sqlite3.Error as e:
            raise ValueError(f"Failed to connect to SQLite database: {e}")

        # Store attributes in the object
        self.dialect = "SQLite"

        #  Inject the connection object into the global environment to be able to get exposed to other python functions as args.
        self.global_env["conn"] = self._conn

        # Optionally provide a method to close the connection
        def close_connection():
            self._conn.close()

        self.close_connection = close_connection

    def get_sqlite_schema_with_pandas(self) -> str | None:
        """
        Extracts the schema of an SQLite database using pandas.
        """
        try:
            # Query the schema
            schema_query = "SELECT sql FROM sqlite_master WHERE type='table';"
            schema_df = pd.read_sql_query(schema_query, self._conn).to_csv(index=False)
            return schema_df
        except Exception as e:
            print(f"SQLite error: {e}")

    def get_postgres_schema_with_pandas(self) -> pandas.DataFrame | None:
        """
        Extracts the schema of a PostgreSQL database using pandas, exclude unnecessary ones.
        """
        try:
            # Query for tables
            schema_query = """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema NOT IN ('information_schema', 'pg_catalog') AND table_schema NOT LIKE '%%steampipe%%'
            """
            schema_df = pd.read_sql(schema_query, self._conn)
            return schema_df
        except Exception as e:
            print(f"PostgreSQL error: {e}")

    def get_schema(self):
        if self.dialect == "SQLite":
            return self.get_sqlite_schema_with_pandas()
        elif self.dialect == "PostgreSQL":
            return self.get_postgres_schema_with_pandas()
        else:
            raise NotImplementedError("This database hasn't integrated yet.")

    def _is_preservable(self, obj: Any) -> bool:
        """Check if an object should be preserved in its current state."""
        return isinstance(obj, self.PRESERVED_TYPES)

    def _split_environment(self):
        """Split environment into preservable and saveable items."""
        preserved = {}
        saveable = {}

        for key, value in self.global_env.items():
            if self._is_preservable(value):
                preserved[key] = value
            else:
                saveable[key] = value

        return preserved, saveable

    @contextmanager
    def preserve_state(self):
        """Context manager to save and restore global environment state."""
        # Split the environment
        preserved, saveable = self._split_environment()

        # Save the current saveable state
        self.saved_states.append(saveable.copy())

        try:
            yield
        finally:
            if self.saved_states:
                # Restore the previous saveable state
                restored_state = self.saved_states.pop()

                # Clear current global_env
                self.global_env.clear()

                # First, restore the preserved items (connections, etc.)
                self.global_env.update(preserved)

                # Then restore the saved state
                self.global_env.update(restored_state)

    def execute_code(self, code: str) -> str:
        """
        Execute the given Python code in an IPython kernel-type interpreter and return the result.

        Args:
            code (str): The Python code to execute.

        Returns:
            str: Captured output
        """
        if not self.global_env.get("conn", None):
            raise RuntimeError("Database connection is not initialized.")

        output_buffer = io.StringIO()

        with (
            contextlib.redirect_stdout(output_buffer),
            contextlib.redirect_stderr(output_buffer),
        ):
            try:
                # Parse the code into an abstract syntax tree (AST)
                tree = ast.parse(code)
                for node in tree.body:
                    stmt = ast.unparse(node)  # Convert AST node back to source code
                    print(f"In [ ]:\n{stmt}")

                    try:
                        if isinstance(node, ast.Expr):
                            # Evaluate expressions (e.g., x + y, function calls)
                            result = eval(stmt, self.global_env)
                            if result is not None:
                                print(f"Out [ ]:\n{result}")

                        else:
                            # Execute statements (e.g., assignments, loops, function defs)
                            # execute_with_collision_check(stmt, self.global_env)
                            exec(stmt, self.global_env)

                    except Exception as e:
                        self.handle_error(e)

            except SyntaxError as se:
                self.handle_error(se)

        return output_buffer.getvalue()

    def handle_error(self, error):
        """Handle execution errors."""
        print(f"Error: {str(error)}")

    # def connect_to_postgres(
    #         self,
    #         host: Union[str, None],
    #         dbname: Union[str, None],
    #         user: Union[str, None],
    #         password: Union[str, None],
    #         port: Union[int, None],
    #         **kwargs,
    # ) -> None:
    #     """
    #     Connect to PostgreSQL using SQLAlchemy, which is compatible with pandas.
    #
    #      **Example:**
    #     ```python
    #     quarry.connect_to_postgres(
    #         host="127.0.0.1",
    #         dbname="steampipe",
    #         user="steampipe",
    #         password="password",
    #         port=9193
    #     )
    #     ```
    #     Args:
    #         host (str): The PostgreSQL host.
    #         dbname (str): The PostgreSQL database name.
    #         user (str): The PostgreSQL user.
    #         password (str): The PostgreSQL password.
    #         port (int): The PostgreSQL port.
    #     """
    #     try:
    #         from sqlalchemy import create_engine
    #         import pandas as pd
    #     except ImportError:
    #         raise DependencyError(
    #             "You need to install required dependencies to execute this method. "
    #             "Run command: pip install quarry[postgres] sqlalchemy"
    #         )
    #
    #     # Fallback to environment variables
    #     host = host or os.getenv("HOST")
    #     dbname = dbname or os.getenv("DATABASE")
    #     user = user or os.getenv("PG_USER")
    #     password = password or os.getenv("PASSWORD")
    #     port = port or int(os.getenv("PORT", 5432))
    #
    #     # Validate required parameters
    #     if not all([host, dbname, user, password, port]):
    #         raise ImproperlyConfigured(
    #             "Missing required PostgreSQL configuration parameters."
    #         )
    #
    #     # Create SQLAlchemy connection string
    #     connection_string = f"postgresql+psycopg://{user}:{password}@{host}:{port}/{dbname}"
    #
    #     # Create SQLAlchemy engine
    #     try:
    #         logger.info("Creating SQLAlchemy engine...")
    #         engine = create_engine(connection_string)
    #         self._conn = engine.connect()
    #         logger.info("Successfully established database connection")
    #
    #     except Exception as e:
    #         raise ValidationError(f"Failed to connect to PostgreSQL: {e}")
    #
    #     # Store attributes in the object
    #     self.dialect = "PostgreSQL"
    #     self.global_env["conn"] = engine
    #
    #     # Optionally provide a method to close the connection
    #     def close_connection():
    #         self._conn.close()
    #         engine.dispose()
    #
    #     self.close_connection = close_connection

    def connect_to_postgres(
        self,
        host: Union[str, None],
        dbname: Union[str, None],
        user: Union[str, None],
        password: Union[str, None],
        port: Union[int, None],
        **kwargs,
    ) -> None:
        """
        Connect to PostgreSQL using the psycopg3 connector.

        **Example:**
        ```python
        quarry.connect_to_postgres(
            host="127.0.0.1",
            dbname="steampipe",
            user="steampipe",
            password="password",
            port=9193
        )
        ```
        Args:
            host (str): The PostgreSQL host.
            dbname (str): The PostgreSQL database name.
            user (str): The PostgreSQL user.
            password (str): The PostgreSQL password.
            port (int): The PostgreSQL port.
        """
        try:
            from psycopg import connect, Error
        except ImportError:
            raise DependencyError(
                "You need to install required dependencies to execute this method. "
                "Run command: pip install quarry[postgres]"
            )

        # Fallback to environment variables
        host = host or os.getenv("DB_HOST")
        dbname = dbname or os.getenv("DB_NAME")
        user = user or os.getenv("DB_USER")
        password = password or os.getenv("DB_PASSWORD")
        port = port or int(os.getenv("PORT", 9193))

        # Validate required parameters
        if not all([host, dbname, user, password, port]):
            raise ImproperlyConfigured(
                "Missing required PostgreSQL configuration parameters."
            )

        # Connection function
        try:
            self._conn = connect(
                host=host,
                dbname=dbname,
                user=user,
                password=password,
                port=port,
                **kwargs,
            )
        except Error as e:
            raise ValidationError(f"Failed to connect to PostgreSQL: {e}")

        # Store attributes in the object
        self.dialect = "PostgreSQL"
        #  Inject the connection object into the global environment to be able to get exposed to other python functions as args.
        self.global_env["conn"] = self._conn

        # Optionally provide a method to close the connection
        def close_connection():
            self._conn.close()

        self.close_connection = close_connection
