"""Install script for setuptools."""

from setuptools import find_packages, setup

# Get the version from metadata.
# version = _metadata.__version__  # type: ignore

long_description = """Smart Trading Bot is an application of reinforcement
learning to financial trading. It is intended as a fun project for
experimenting with financial applications of modern RL."""

testing_formatting_requirements = [
    "pre-commit",
    "mypy==0.812",
    "pytest-xdist",
    "flake8==3.9.1",
    "black==21.4b1",
    "pytest-cov",
]

tf_requirements = ["tensorflow>=2.4.0"]
yf_requirements = ["yfinance>=0.1.59"]

setup(
    name="id-SmartTrainerBot",
    # version=version,
    description="Smart Trading Bot is an application of RL to financial trading",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="InstaDeep",
    license="Apache License, Version 2.0",
    keywords="finance machine learning reinforcement learning",
    packages=find_packages(),
    install_requires=["numpy"],
    extras_require={
        "tf": tf_requirements,
        "yf": yf_requirements,
        "testing_formatting": testing_formatting_requirements,
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
