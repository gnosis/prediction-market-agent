from pathlib import Path

from modal import Image

this_file_abs_path = __file__
pyproject_toml = Path(this_file_abs_path).parent.parent.parent.parent / "pyproject.toml"

MODAL_IMAGE = Image.debian_slim().poetry_install_from_file(
    poetry_pyproject_toml=pyproject_toml
)
