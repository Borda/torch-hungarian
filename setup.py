"""Package configuration for the pure-source distribution."""

import setuptools


def get_requirements() -> list[str]:
    """Read install dependencies without retaining blank requirement entries."""
    with open("requirements.txt", encoding="utf-8") as requirements_file:
        return [line.strip() for line in requirements_file if line.strip()]


def get_long_description() -> str:
    """Read the package long description from the repository README."""
    with open("README.md", encoding="utf-8") as readme_file:
        return readme_file.read()


if __name__ == "__main__":
    setuptools.setup(
        name="torch-hungarian",
        version="0.1.0rc1",
        author="Ivan Karpukhin",
        author_email="karpuhini@yandex.ru",
        maintainer="Jirka Borovec",
        maintainer_email="j.borovec@gmail.com",
        description="Batched linear assignment with PyTorch and CUDA.",
        long_description=get_long_description(),
        long_description_content_type="text/markdown",
        url="https://github.com/Borda/torch-hungarian",
        project_urls={
            "Source": "https://github.com/Borda/torch-hungarian",
            "Changelog": "https://github.com/Borda/torch-hungarian/blob/main/CHANGELOG.md",
            "Upstream": "https://github.com/ivan-chai/torch-linear-assignment",
        },
        packages=["torch_linear_assignment"],
        python_requires=">=3.10",
        install_requires=get_requirements(),
    )
