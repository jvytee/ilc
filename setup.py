from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_desc = f.read()

setup(
    name="ILC",
    version="0.1.0",
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "scikit-learn"
    ],
    author="Julian Theis",
    author_email="julian.theis@posteo.de",
    description="Simple implementation of an Iterative Learning Controller",
    url="https://github.com/jvytee/ilc"
)
