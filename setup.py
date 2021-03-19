from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="transquest",
    version="1.0.0-beta",
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
        "numpy==1.19.0",
        "requests==2.24.0",
        "tqdm==4.47.0",
        "regex==2020.6.8",
        "transformers==3.0.1",
        "scipy==1.5.1",
        "scikit-learn==0.23.1",
        "tensorboardx==2.1",
        "pandas==1.0.5",
        "tokenizers==0.8.0",
        "matplotlib==3.2.2",
        "wandb==0.9.2",
        "googledrivedownloader==0.4"
    ],
)
