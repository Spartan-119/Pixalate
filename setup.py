from setuptools import setup, find_packages

# Function to read the requirements.txt file
def read_requirements():
    with open("requirements.txt") as req_file:
        return req_file.read().splitlines()

# Function to read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as readme_file:
        return readme_file.read()

setup(
    name="pixalate",
    version="0.1",
    packages=find_packages(),
    description="A federated learning project for fraud detection",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="Abin",
    url="https://github.com/Spartan-119/Pixalate", 
    install_requires=read_requirements(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)