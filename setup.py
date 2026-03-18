from setuptools import setup, find_packages

setup(
    name="secure-rag",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "sentence-transformers",
        "faiss-cpu",
        "numpy",
        "openai",
        "python-dotenv"
    ],
    author="Atharva Patil",
    description="Privacy-aware RAG framework with masking and streaming support",
)