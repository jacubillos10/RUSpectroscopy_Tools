from setuptools import setup, Extension 
import numpy as np 
import sys

extra_compile_flags = ["-fopenmp"]
extra_link_flags = ["-fopenmp"]

setup(
        name = "ruspectroscopy_tools",
        version = "0.0.1",
        description = "A simple tool that generates Gamma matrix and E matrix, given the elastic constants, mass, dimansions and shape of a sample, in Resonant Ultrasound Spectroscopy. Use scipy.linalg.eigh(...) to solve the forward problem",
        long_description = open("README.md").read(),
        long_description_content_type = "text/markdown",
        author = "Alejandro Cubillos",
        author_email = "alejandro4cm@outlook.com",
        url = "https://github.com/jacubillos10/RUSpectroscopy_Tools",
        packages = ["rusmodules"],
        ext_modules =  [
            Extension(
                "rusmodules.rus",
                sources = ["rusmodules/rus.c"],
                include_dirs = [np.get_include()],
                extra_compile_args = extra_compile_flags,
                extra_link_args = extra_link_flags,
            )
        ],
        classifiers = [
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
        ],
        license = "MIT",
        python_requires = ">=3.6",
        setup_requires = ["numpy"],
)
