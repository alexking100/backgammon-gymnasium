from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="backgammon-gymnasium",
    version="1.0.0",
    author="Alex King",
    author_email="your-email@example.com",  # Update with your email
    description="A modern backgammon environment compatible with Gymnasium",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/alexking100/backgammon-gymnasium",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Games/Entertainment :: Board Games",
    ],
    python_requires=">=3.7",
    install_requires=[
        "gymnasium>=0.28.0",
        "numpy",
        "pyglet",
        "matplotlib",
        "pandas",
    ],
    extras_require={
        "dev": [
            "pytest",
            "pytest-cov",
            "black",
            "flake8",
        ],
    },
    entry_points={
        "console_scripts": [
            "backgammon-random=examples.play_random_agent:main",
        ],
    },
)