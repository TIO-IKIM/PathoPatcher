from setuptools import find_packages, setup

VERSION = "0.9.3a"
DESCRIPTION = "PathoPatch - Accelerating Artificial Intelligence Based Whole Slide Image Analysis with an Optimized Preprocessing Pipeline"
with open("docs/README_pypi.md", "r") as fh:
    LONG_DESCRIPTION = fh.read()

# Setting up
setup(
    # the name must match the folder name 'verysimplemodule'
    name="pathopatch",
    version=VERSION,
    author="Fabian Hörst",
    author_email="fabian.hoerst@uk-essen.de",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    url="https://github.com/TIO-IKIM/PathoPatcher",
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=["tests", "tests.*"]),
    python_requires=">=3.9",
    install_requires=[
        "Pillow>=9.5.0",
        "PyYAML",
        "Shapely==1.8.5.post1",
        "colorama",
        "future",
        "geojson>=3.0.0",
        "matplotlib",
        "natsort",
        "numpy>1.22,<1.24",
        "opencv_python_headless",
        "openslide_python",
        "pandas",
        "pydantic==1.10.4",
        "rasterio==1.3.5.post1",
        "requests",
        "scikit-image",
        "setuptools<=65.6.3",
        "tqdm",
        "torchvision",
        "torch",
    ],
    scripts=[
        "pathopatch/wsi_extraction.py",
        "pathopatch/annotation_conversion.py",
        "pathopatch/macenko_vector_generation.py",
    ],
    entry_points={
        "console_scripts": [
            "wsi_extraction=pathopatch.wsi_extraction:main",
            "annotation_conversion=pathopatch.annotation_conversion:main",
            "macenko_vector_generation=pathopatch.macenko_vector_generation:main",
        ],
    },
    keywords=["python", "pathopatch"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Other",
    ],
)
