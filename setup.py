"""A setuptools based setup module.
See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

from setuptools import find_packages, setup

setup(
    name="net_benefit_ascvd",
    version="0.0.1",
    description="",
    url="https://github.com/net_benefit_ascvd",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas>=1.0.0",
        "torch>=1.6",
        "pyyaml",
        "pyarrow",
        "scipy",
        "sklearn",
        "configargparse",
        "matplotlib",
        "sqlalchemy",
        "dask>=2.14.0",
        "google-cloud-bigquery",
        "google-cloud-bigquery-storage",
        "pandas-gbq",
        "pytest",
        "tqdm",
        "joblib"
    ],
)
