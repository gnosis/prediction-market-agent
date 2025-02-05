#!/bin/bash

# Install pre-commit if not already installed
if ! command -v pre-commit &> /dev/null
then
    echo "pre-commit is not installed. Installing now..."
    brew install pre-commit
fi

# Install the hooks
pre-commit install

echo "Pre-commit hooks installed successfully."
