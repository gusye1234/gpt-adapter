import setuptools

with open("readme.md", "r") as fh:
    long_description = fh.read()


vars2find = ["__author__", "__version__", "__url__"]
vars_project = {}
with open("./gpt_adapter/__init__.py") as f:
    for line in f.readlines():
        for v in vars2find:
            if line.startswith(v):
                line = line.replace(" ", "").replace('"', "").replace("'", "").strip()
                vars_project[v] = line.split("=")[1]

setuptools.setup(
    name="gpt_adapter",
    url=vars_project["__url__"],
    version=vars_project["__version__"],
    author=vars_project["__author__"],
    description="Collections of algorithms for efficiently fine-tuning LLM",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=["gpt_adapter"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=["torch", "transformers"],
)
