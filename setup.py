from setuptools import setup, find_packages

setup(
    name="secure-rag",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pypdf",
        "sentence-transformers",
        "faiss-cpu",
        "numpy",
        "openai",
        "python-dotenv",
        "typer",
        "rich"
    ],
    entry_points={
        "console_scripts": [
            "secure-rag=secure_rag.cli:main"
        ]
        
    },
    author="Atharva Patil",
    description="Privacy-aware RAG framework with masking and streaming support",
)
