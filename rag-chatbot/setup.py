from setuptools import setup, find_packages

setup(
    name="rag-chatbot",
    version="0.1.0",
    description="Production-quality End-to-End Retrieval-Augmented Generation Chatbot",
    author="ML Portfolio",
    python_requires=">=3.9",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=2.0.0",
        "sentence-transformers>=2.2.0",
        "faiss-cpu>=1.7.4",
        "transformers>=4.30.0",
        "gradio>=3.35.0",
        "pypdf>=3.0.0",
        "scikit-learn>=1.3.0",
        "numpy>=1.24.0",
        "openai>=1.0.0",
        "tqdm>=4.65.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
