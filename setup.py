from pathlib import Path
from setuptools import setup, find_packages

HERE = Path(__file__).parent

# read long description from README
long_description = (HERE / "README.md").read_text(encoding="utf-8") if (HERE / "README.md").exists() else ""

# parse requirements.txt if present
def parse_requirements(path="requirements.txt"):
    reqs = []
    p = HERE / path
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            reqs.append(line)
    return reqs

setup(
    name="paw_statistics",
    version="0.0.0",  # bump or use setuptools_scm (see notes below)
    description="Python framework for analysis of static hind paw postures",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="ChristianPritz",
    packages=find_packages(exclude=("tests", "docs")),
    include_package_data=True,
    install_requires=parse_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov",
            "flake8",
            "black",
            "ruff",
        ],
        "all": [],
    },
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
