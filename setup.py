from setuptools import setup

setup(
    name="feature_selector",
    version="0.1",
    author="Dinesh Mannari",
    author_email="dmannari@iu.edu",
    description="A package for feature selection",
    packages=["feature_selector"],
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn"
    ]
)
