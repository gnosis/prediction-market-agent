from setuptools import setup, find_packages

# Read requirements from requirements.txt
with open("requirements.txt", "r") as f:
    install_requires = f.read().splitlines()

# Remove any comments or empty lines
install_requires = [
    req.strip() for req in install_requires if req and not req.startswith("#")
]

setup(
    name="prediction_market_agent",
    version="0.1",
    packages=find_packages(),
    install_requires=install_requires,
)
