from pathlib import Path

from setuptools import find_packages, setup

BASE_DIR = Path(__file__).parent
README = (BASE_DIR / "README.md").read_text(encoding="utf-8")

setup(
    name="secure-rag",
    version="0.1.0",
    description="Privacy-aware RAG framework with masking and streaming support",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Atharva Patil",
    license="MIT",
    url="https://github.com/Atharvapatil2005/RAG",
    packages=find_packages(exclude=("test", "test.*")),
    include_package_data=True,
    install_requires=[
        "pypdf",
        "sentence-transformers",
        "faiss-cpu",
        "numpy",
        "openai",
        "python-dotenv",
        "typer",
        "rich",
    ],
    entry_points={
        "console_scripts": [
            "secure-rag=secure_rag.cli:main",
        ]
    },
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Security",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
