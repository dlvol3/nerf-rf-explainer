from setuptools import setup, find_packages

setup(
    name="nerf-rf-explainer",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "black",
        "numpy",
        "pandas",
        "jupyter"
    ],
    author="Yue",
    description="A Python toolbox project initialized by wutuready 🐱"
)
