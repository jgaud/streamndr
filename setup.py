import os
import setuptools

# Meta-Data
NAME = "streamndr"
DESCRIPTION = "Stream Novelty Detection for River"
LONG_DESCRIPTION_CONTENT_TYPE = "text/markdown"
URL = "https://jgaud.github.io/streamndr/"
EMAIL = "jean-gabrielgaudreault@outlook.com"
AUTHOR = "Jean-Gabriel Gaudreault"
REQUIRES_PYTHON = ">=3.9.0"

# Requirements
base_packages = [
    "scikit-learn>=1.2.1",
    "pandas>=1.4.2",
    "numpy>=1.23.5",
    "river>=0.21.0",
    "clusopt-core>=1.0.0"
]

here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = "\n" + f.read()

about = {}
with open(os.path.join(here, NAME, "__version__.py")) as f:
    exec(f.read(), about)

setuptools.setup(
    name=NAME,
    version=about["__version__"],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type=LONG_DESCRIPTION_CONTENT_TYPE,
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=setuptools.find_packages(exclude=("tests",)),
    install_requires=base_packages,
    include_package_data=True,
    license="BSD-3",
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "License :: OSI Approved :: BSD License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: Implementation :: PyPy",
    ],
    ext_modules=[],
)