from setuptools import setup, find_packages

setup(
    name="picasa",
    version="0.1.0",
    description="PICASA, Partitioning Inter-patient Cellular Attributes by Shared Attention, project python package",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Sishir Subedi and Yongjin Park",
    author_email="",
    url="https://github.com/causalpathlab/picasa",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "anndata==0.10.8",
        "annoy==1.17.0",
        "numpy==1.24.4",
        "pandas>=2.0.3",
        "scanpy==1.9.3",
        "torch==2.5.1"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
