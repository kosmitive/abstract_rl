import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(

    name="abstract_rl",
    version="0.0.1",
    author="",
    author_email="",
    description="A modular reinforcement library based on PyTorch.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.com/kosmitive/abstract_rl",

    packages=setuptools.find_packages(exclude=['docs']),

    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Reinforcement Learning",
        "Intended Audience :: Developers",
        "Development Status :: 3 - Alpha"
    ],

    project_urls={
        "Homepage": "https://wwww.topologicallydisturbed.io/abstract_rl",
        "Source": "https://gitlab.com/kosmitive/abstract_rl"
    },

    keywords='reinforcement learning policy gym'
)

