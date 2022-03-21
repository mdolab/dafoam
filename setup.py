from setuptools import setup, find_packages
import re

__version__ = re.findall(
    r"""__version__ = ["']+([0-9\.]*)["']+""", open("dafoam/pyDAFoam.py").read(),
)[0]

setup(
    name="dafoam",
    version=__version__,
    description="DAFoam: Discrete Adjoint with OpenFOAM for High-fidelity, gradient-based Optimization",
    long_description="""
      DAFoam contains a suite of adjoint solvers to efficiently compute derivatives. It also provides a Python interface to interact with a high-fidelity gradient-based design optimization framework (MACH). DAFoam is based on OpenFOAM and has the following features:

      - It implements an efficient discrete adjoint approach with competitive speed, scalability, accuracy, and compatibility.
      - It allows rapid discrete adjoint development for any steady and unsteady OpenFOAM primal solvers with modifying only a few hundred lines of source codes.
      - It supports design optimizations for a wide range of disciplines such as aerodynamics, heat transfer, structures, hydrodynamics, and radiation.

      """,
    long_description_content_type="text/markdown",
    keywords="OpenFOAM adjoint optimization",
    author="",
    author_email="",
    url="https://github.com/mdolab/dafoam",
    license="GPL version 3",
    packages=find_packages(include=["dafoam*"]),
    package_data={"dafoam": ["*.so"]},
    scripts=[
        "dafoam/scripts/dafoam_matdiff.py",
        "dafoam/scripts/dafoam_vecdiff.py",
        "dafoam/scripts/dafoam_matgetvalues.py",
        "dafoam/scripts/dafoam_vecgetvalues.py",
        "dafoam/scripts/dafoam_plot3d2tecplot.py",
        "dafoam/scripts/dafoam_plot3dtransform.py",
        "dafoam/scripts/dafoam_stltransform.py",
    ],
    install_requires=[
        "numpy>=1.16.4",
        "mpi4py>=3.0.0",
        "petsc4py>=3.11.0",
        "cython>=0.29.21",
    ],
    classifiers=["Operating System :: Linux", "Programming Language :: Cython, C++"],
)
