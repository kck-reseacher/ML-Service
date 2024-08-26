from setuptools import find_packages, setup

setup(
    name="mlops-mls",
    version="1.0",
    description="MLOps-serving",
    author="exem",
    author_email="ex-em.com",
    install_requires=[],
    packages=find_packages(exclude=["docs", "tests*"]),
    python_requires=">=3.8",
    package_data={},
    zip_safe=False,
    classifiers=["Programming Language :: Python :: 3.8.16"],
)
