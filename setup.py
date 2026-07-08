from pathlib import Path

from setuptools import find_packages, setup

__lib_name__ = "DVCAlign"
__lib_version__ = "1.0.0"
__description__ = "Integrating spatial transcriptomics data across different conditions, technologies, and developmental stages"
__url__ = "https://github.com/VitaIntelli-CQU/DVCAlign"
__author__ = "Cheng Wei"
__author_email__ = "2804775192@qq.com"
__license__ = "All rights reserved"
__keywords__ = ["spatial transcriptomics", "data integration", "Graph attention auto-encoder", "spatial domain", "three-dimensional reconstruction"]


def _read_requirements(filename: str):
    requirement_path = Path(__file__).with_name(filename)
    return [
        line.strip()
        for line in requirement_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


__requires__ = _read_requirements("requirement.txt")

setup(
    name=__lib_name__,
    version=__lib_version__,
    description=__description__,
    url=__url__,
    author=__author__,
    author_email=__author_email__,
    license=__license__,
    packages=find_packages(include=["DVCAlign", "DVCAlign.*"]),
    install_requires=__requires__,
    python_requires=">=3.8",
    zip_safe=False,
    include_package_data=True,
)
