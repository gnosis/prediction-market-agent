import json
import tempfile

from autogen.coding import CodeBlock, LocalCommandLineCodeExecutor
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
            '[{"language":"bash","code":"pip install pmat"},{"language":"python","code":"print(\'Hello, World!\')"}]'
        ]

    def __call__(self, code_blocks: str) -> str:
        code_blocks_parsed = [
            CodeBlock.model_validate(b) for b in json.loads(code_blocks)
        ]

        # Simple check to forbid out any malicious code.
        for code_block in code_blocks_parsed:
            LocalCommandLineCodeExecutor.sanitize_command(
                code_block.language, code_block.code
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            executor = LocalCommandLineCodeExecutor(
                timeout=60,
                work_dir=temp_dir,
            )
            result = executor.execute_code_blocks(code_blocks_parsed)

        return f"Code finished with exit code {result.exit_code} and output:\n\n{result.output}"


CODE_FUNCTIONS: list[type[Function]] = [
    ExecuteCodeFunction,
]
