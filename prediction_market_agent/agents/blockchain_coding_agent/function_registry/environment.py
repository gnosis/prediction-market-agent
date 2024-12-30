import inspect
import json
import sqlite3
from typing import Callable, List, Dict, Type


def serialize_type(py_type):
    """
    Converts a Python type to a string representation for storage.
    """
    return py_type.__name__ if py_type else None


def deserialize_type(type_str):
    """
    Converts a string representation back to a Python type.
    """
    if not type_str:
        return None
    return {
        "int": int,
        "float": float,
        "str": str,
        "bool": bool,
        "dict": dict,
        "list": list,
    }.get(
        type_str, None
    )  # Add more types as needed


class FunctionDefinition:
    def __init__(
        self,
        name: str,
        code: str,
        dependencies: List[str],
        input_types: Dict[str, Type],
        output_type: Type,
    ):
        self.name = name
        self.code = code
        self.dependencies = dependencies
        self.input_types = input_types
        self.output_type = output_type
        self.compiled_function: Callable = None


class FunctionRegistry:
    def __init__(self, db_path="functions.db", global_dependencies=None):
        """
        Initialize the FunctionRegistry.
        :param global_dependencies: A dictionary of global dependencies (e.g., external libraries) to inject into the namespace.
        """
        self.functions = {}
        self.global_dependencies = global_dependencies or {}
        self.db_path = db_path

    def _connect_db(self):
        conn = sqlite3.connect(self.db_path)
        return conn

    def save(self):
        conn = self._connect_db()
        cursor = conn.cursor()

        # Create table if not exists
        cursor.execute(
            """
               CREATE TABLE IF NOT EXISTS functions (
                   name TEXT PRIMARY KEY,
                   code TEXT,
                   dependencies TEXT,
                   input_types TEXT,
                   output_type TEXT
               )
           """
        )

        # Insert functions
        for func_name, func in self.functions.items():
            cursor.execute(
                """
                INSERT OR REPLACE INTO functions (name, code, dependencies, input_types, output_type)
                VALUES (?, ?, ?, ?, ?)
            """,
                (
                    func.name,
                    func.code,
                    ",".join(func.dependencies),
                    json.dumps(
                        {k: serialize_type(v) for k, v in func.input_types.items()}
                    ),  # Serialize input types
                    serialize_type(func.output_type),  # Serialize output type
                ),
            )

        conn.commit()
        conn.close()

    def load(self):
        conn = self._connect_db()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM functions")
        rows = cursor.fetchall()

        for row in rows:
            func_name, code, dependencies, input_types, output_type = row
            dependencies = dependencies.split(",")
            input_types = {
                k: deserialize_type(v) for k, v in json.loads(input_types).items()
            }
            output_type = deserialize_type(output_type)

            function_def = FunctionDefinition(
                name=func_name,
                code=code,
                dependencies=dependencies,
                input_types=input_types,
                output_type=output_type,
            )
            self.functions[func_name] = function_def

        conn.close()

    def add_function(self, function_def: FunctionDefinition):
        # Validate dependencies
        for dependency in function_def.dependencies:
            if dependency not in self.functions:
                raise ValueError(f"Dependency '{dependency}' not found in registry.")

        # Compile function
        namespace = {
            dep: self.functions[dep].compiled_function
            for dep in function_def.dependencies
        }
        namespace.update(
            self.global_dependencies
        )  # Inject global dependencies (e.g., NumPy)
        exec(function_def.code, namespace)

        # Extract the function and validate its signature
        func = namespace[function_def.name]
        sig = inspect.signature(func)

        # Check input types
        for param_name, param in sig.parameters.items():
            expected_type = function_def.input_types.get(param_name)
            if expected_type and param.annotation != expected_type:
                raise TypeError(
                    f"Parameter '{param_name}' must be of type {expected_type}, got {param.annotation}."
                )

        # Check output type
        if sig.return_annotation != function_def.output_type:
            raise TypeError(
                f"Return type must be {function_def.output_type}, got {sig.return_annotation}."
            )

        # Store in registry
        function_def.compiled_function = func
        self.functions[function_def.name] = function_def

    def resolve_execution_order(self):
        from collections import defaultdict, deque

        graph = defaultdict(list)
        indegree = defaultdict(int)

        # Build dependency graph
        for func in self.functions.values():
            for dep in func.dependencies:
                graph[dep].append(func.name)
                indegree[func.name] += 1

        # Topological sort
        queue = deque([name for name in self.functions if indegree[name] == 0])
        order = []

        while queue:
            current = queue.popleft()
            order.append(current)
            for neighbor in graph[current]:
                indegree[neighbor] -= 1
                if indegree[neighbor] == 0:
                    queue.append(neighbor)

        if len(order) != len(self.functions):
            raise RuntimeError("Cyclic dependency detected.")

        return order


class FunctionExecutor:
    def __init__(self, registry: FunctionRegistry):
        self.registry = registry

    def execute_all(self):
        """
        Execute all functions in the registry in topological order.
        """
        order = self.registry.resolve_execution_order()
        results = {}
        for func_name in order:
            func_def = self.registry.functions[func_name]
            results[func_name] = func_def.compiled_function()
        return results

    def execute_function(self, function_name: str):
        """
        Execute a specific function by name, resolving its dependencies.
        """
        if function_name not in self.registry.functions:
            raise ValueError(f"Function '{function_name}' is not in the registry.")

        # Resolve execution order, focusing only on the target function and its dependencies
        resolved_order = self._resolve_dependencies(function_name)
        results = {}
        for func_name in resolved_order:
            func_def = self.registry.functions[func_name]
            results[func_name] = func_def.compiled_function()

        # Return the result of the specific function requested
        return results[function_name]

    def _resolve_dependencies(self, target_function: str):
        """
        Perform a dependency resolution for the target function.
        """
        from collections import defaultdict, deque

        # Build dependency graph for all functions in the registry
        graph = defaultdict(list)
        reverse_graph = defaultdict(list)
        for func in self.registry.functions.values():
            for dep in func.dependencies:
                graph[dep].append(func.name)
                reverse_graph[func.name].append(dep)

        # Perform a reverse BFS to find all dependencies of the target function
        visited = set()
        queue = deque([target_function])
        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)
            for dep in reverse_graph[current]:
                queue.append(dep)

        # Topologically sort the visited functions
        order = []
        indegree = {func: 0 for func in visited}
        for func in visited:
            for neighbor in graph[func]:
                if neighbor in visited:
                    indegree[neighbor] += 1

        queue = deque([func for func in visited if indegree[func] == 0])
        while queue:
            current = queue.popleft()
            order.append(current)
            for neighbor in graph[current]:
                if neighbor in visited:
                    indegree[neighbor] -= 1
                    if indegree[neighbor] == 0:
                        queue.append(neighbor)

        if len(order) != len(visited):
            raise RuntimeError(
                f"Cyclic dependency detected while resolving '{target_function}'."
            )

        return order
