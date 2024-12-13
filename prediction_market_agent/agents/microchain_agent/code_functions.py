import asyncio
import json
import tempfile

from autogen_core import CancellationToken
from autogen_core.code_executor import CodeBlock
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
from microchain import Function


class ExecuteCodeFunction(Function):
    @property
    def description(self) -> str:
        return (
            f"Use this function to execute arbitrary code in any of the following languages {LocalCommandLineCodeExecutor.SUPPORTED_LANGUAGES}."
            " Input is a signle json encoded list of code blocks to execute. Each code block needs to have a 'language' and 'code' field."
            " To get any output, simply print the variables you want to see."
            " To install non existing libraries, use `bash` language and use the language's package tool to install the library."
            " Prepend the code block with library installation command before the actual code block and then try again."
        )

    @property
    def example_args(self) -> list[str]:
        return [
            '[{"language":"bash","code":"pip install prediction-market-agent-tooling"},{"language":"python","code":"print(\'Hello, World!\')"}]'
        ]

    def __call__(self, code_blocks: str) -> str:
        code_blocks_parsed = [
            CodeBlock(**loaded_code_block)
            for loaded_code_block in json.loads(code_blocks)
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            executor = LocalCommandLineCodeExecutor(
                timeout=60,
                work_dir=temp_dir,
            )
            # We assume no event loop is currently running.
            result = asyncio.run(
                executor.execute_code_blocks(
                    code_blocks=code_blocks_parsed,
                    cancellation_token=CancellationToken(),
                )
            )

        return f"Code finished with exit code {result.exit_code} and output:\n\n{result.output}"


CODE_FUNCTIONS: list[type[Function]] = [
    ExecuteCodeFunction,
]
