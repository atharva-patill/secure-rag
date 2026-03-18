from setuptools import setup, find_packages

setup(
    name="secure-rag",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "faiss-cpu",
        "sentence-transformers",
        "openai",
        "python-dotenv",
    ],
)