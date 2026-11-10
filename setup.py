from setuptools import setup, find_namespace_packages


setup(
    name="mobipi",
    version="0.1.0",
    description="A library for mobilizing non-mobile robot policies",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Jingyun Yang",
    author_email="jingyuny@stanford.edu",
    url="https://github.com/yjy0625/mobipi",
    packages=find_namespace_packages(
        include=("mobipi", "mobipi.*", "rby1", "rby1.*"),
        exclude=("*.__pycache__",),
    ),
    package_data={
        "rby1": ["configs/*.json", "configs/mimicgen/*.json", "tests/fixtures/*.json"],
        "mobipi.collect_demos": ["*.json"],
    },
    install_requires=[
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
    ],
    keywords="robotics",
    python_requires=">=3.7",
)
