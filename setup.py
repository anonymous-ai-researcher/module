"""
NFMR: Noise-Free Module Retrieval
Package Setup Configuration
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
if requirements_path.exists():
    requirements = [
        line.strip() 
        for line in requirements_path.read_text().splitlines() 
        if line.strip() and not line.startswith("#")
    ]
else:
    requirements = []

setup(
    name="nfmr",
    version="0.1.0",
    author="SIGIR 2026 Authors",
    author_email="author@example.com",
    description="Noise-Free Module Retrieval from Large-Scale Knowledge Bases",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/nfmr",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/nfmr/issues",
        "Documentation": "https://github.com/yourusername/nfmr#readme",
        "Source Code": "https://github.com/yourusername/nfmr",
    },
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    keywords=[
        "knowledge-base",
        "ontology",
        "forgetting",
        "module-extraction",
        "RAG",
        "retrieval-augmented-generation",
        "description-logic",
        "semantic-web",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.21.0",
        "networkx>=2.6.0",
        "rdflib>=6.0.0",
        "pyyaml>=6.0",
        "psutil>=5.8.0",
        "python-dotenv>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "mypy>=0.950",
            "flake8>=4.0.0",
        ],
        "llm": [
            "torch>=2.0.0",
            "transformers>=4.30.0",
            "openai>=1.0.0",
            "anthropic>=0.18.0",
            "tiktoken>=0.5.0",
        ],
        "viz": [
            "matplotlib>=3.5.0",
            "seaborn>=0.12.0",
            "plotly>=5.10.0",
        ],
        "all": [
            "torch>=2.0.0",
            "transformers>=4.30.0",
            "openai>=1.0.0",
            "anthropic>=0.18.0",
            "tiktoken>=0.5.0",
            "matplotlib>=3.5.0",
            "seaborn>=0.12.0",
            "plotly>=5.10.0",
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "nfmr-benchmark=nfmr.experiments.run_benchmark:main",
            "nfmr-rag-eval=nfmr.experiments.run_rag_eval:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
