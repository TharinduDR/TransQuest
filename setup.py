from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="transquest",
    version="1.1.1",
    author="Tharindu Ranasinghe",
    author_email="rhtdranasinghe@gmail.com",
    description="Transformer based translation quality estimation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/TharinduDR/TransQuest",
    packages=find_packages(exclude=("examples", "docs", )),
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.6",
    install_requires=[
        "numpy",
        "requests",
        "tqdm>=4.47.0",
        "regex",
        "transformers>=4.2.0",
        "scipy",
        "scikit-learn",
        "tensorboardx",
        "pandas",
        "tokenizers",
        "matplotlib",
        "wandb",
        "sentencepiece",
        "onnxruntime",
        "seqeval",
    ],
)
