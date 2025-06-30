import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="trim-transformer",
    version="0.1.0",
    author="Emanuel Gordis",
    author_email="emanuel@nuclearsoftware.com",
    description="A linear-attention transformer implementation with KV caching.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/emanuel-nuclearsoftware/trim-transformer",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        'torch',
    ],
)
