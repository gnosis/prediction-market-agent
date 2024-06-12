from langchain.tools import BaseTool
from microchain import Function


def microchain_function_from_tool(tool: BaseTool, example_args: list[str]) -> Function:
    class NewFunction(Function):
        __call__ = tool._run
        name = tool.name

    f = NewFunction()
    f.example_args = example_args
    f.description = tool.description
    return f
