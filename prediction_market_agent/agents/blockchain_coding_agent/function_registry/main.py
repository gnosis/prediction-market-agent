# Initialize registry
import numpy

from prediction_market_agent.agents.blockchain_coding_agent.function_registry.environment import (
    FunctionExecutor,
    FunctionDefinition,
    FunctionRegistry,
)

registry = FunctionRegistry(global_dependencies={"np": numpy})
registry.save()
registry.load()

# Add a base function
base_function_code = """
def base_function() -> int:
    return 42
"""

base_function_def = FunctionDefinition(
    name="base_function",
    code=base_function_code,
    dependencies=[],
    input_types={},
    output_type=int,
)

registry.add_function(base_function_def)

# Add a dependent function
dependent_function_code = """
def dependent_function() -> str:
    result = base_function()
    return f"Result is {result}"
"""


dependent_function_def = FunctionDefinition(
    name="dependent_function",
    code=dependent_function_code,
    dependencies=["base_function"],
    input_types={},
    output_type=str,
)

registry.add_function(dependent_function_def)

# Add a function that depends on numpy
numpy_function_code = """
def random_number_function() -> float:
    return np.random.random()
"""

numpy_function_def = FunctionDefinition(
    name="random_number_function",
    code=numpy_function_code,
    dependencies=[],  # No function dependencies, but relies on NumPy
    input_types={},
    output_type=float,
)

registry.add_function(numpy_function_def)

# Execute functions
executor = FunctionExecutor(registry)

# Execute a specific function
dependent_result = executor.execute_function("dependent_function")
print("Dependent function result:", dependent_result)

# Execute the NumPy-dependent function
random_number_result = executor.execute_function("random_number_function")
print("Random number function result:", random_number_result)
registry.save()
