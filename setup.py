from pathlib import Path
from setuptools import find_namespace_packages, setup

root = Path(__file__).parent
requirements = [
    line.strip()
    for line in (root / "requirements.txt").read_text().splitlines()
    if line.strip() and not line.lstrip().startswith("#")
]

setup(
    name="raptgfn",
    version="0.1.0",
    description="Preference-conditioned RNA aptamer candidate generation",
    long_description=(root / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    license="MIT",
    packages=find_namespace_packages(where="src", include=["raptgfn*"]),
    package_dir={"": "src"},
    package_data={"raptgfn": ["configs/*.yaml", "configs/*/*.yaml"]},
    install_requires=requirements,
    python_requires=">=3.10",
    entry_points={"console_scripts": ["raptgfn=raptgfn.cli:main"]},
)
