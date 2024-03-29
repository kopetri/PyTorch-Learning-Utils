import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pytorch_utils",
    version="1.0.0",
    author="Sebastian Hartwig",
    author_email="sebastian.hartwig@uni-ulm.de",
    description="Collection of functions for PyTorch.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        'torch>=2.0.0',
        'lightning>=2.0.0',
        'wandb'
    ]
)