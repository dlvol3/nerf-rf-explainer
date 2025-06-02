from setuptools import setup, find_packages

setup(
    name="nerf-rf-explainer",
    version="0.1.1",
    packages=find_packages(),
    url="https://github.com/dlvol3/nerf-rf-explainer",
    install_requires=[
        "numpy>=1.20",
        "pandas>=1.3",
        "scikit-learn>=1.0",
        "networkx>=2.6",
        "matplotlib>=3.4",
        "seaborn>=0.11",
        "tqdm>=4.60",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    author="Yue Zhang",
    description="NERF: Interpreting Random Forest with Ensemble Networks for Omics Data Analysis",
)
