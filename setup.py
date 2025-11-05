"""
Setup script for AdaptiveGenieNetwork
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="adaptive-genie-network",
    version="1.0.0",
    author="AdaptiveGenieNetwork Development Team",
    author_email="contact@adaptivegenienetwork.org",
    description="A revolutionary optimization framework using dialectical parameter adaptation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/adaptivegenienetwork/adaptive-genie-network",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.12.0",
            "black>=21.0.0",
            "flake8>=3.9.0",
            "mypy>=0.910",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "sphinx-autodoc-typehints>=1.12.0",
        ],
        "jupyter": [
            "jupyter>=1.0.0",
            "ipywidgets>=7.6.0",
            "jupyterlab>=3.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "agn-demo=demo:main",
        ],
    },
    keywords="optimization, dialectical, swarm intelligence, evolutionary computation, adaptive algorithms",
    project_urls={
        "Bug Reports": "https://github.com/adaptivegenienetwork/adaptive-genie-network/issues",
        "Source": "https://github.com/adaptivegenienetwork/adaptive-genie-network",
        "Documentation": "https://adaptive-genie-network.readthedocs.io/",
    },
)