'''

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v1.1

    Description:
        Cython setup file for wrapping OpenFOAM libraries and solvers.
        One needs to set include dirs/files and flags according to the 
        information in Make/options and Make/files in OpenFOAM libraries 
        and solvers. Then, follow the detailed instructions below. The 
        python naming convention is to add "py" before the C++ class name
'''

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import os, petsc4py

libName="pyBuoyantSimpleDAFoam"

os.environ["CC"] = "mpicc" 
os.environ["CXX"] = "mpicxx"

# These setup should reproduce calling wmake to compile OpenFOAM libraries and solvers
ext = [Extension(libName,
    # All source files, taken from Make/files
    sources=[
        "pyBuoyantSimpleDAFoam.pyx",
        "BuoyantSimpleDAFoam.C"],
    # All include dirs, refer to Make/options in OpenFOAM
    include_dirs=[
        # These are from Make/options:EXE_INC 
        os.getenv("FOAM_SRC")+"/transportModels/compressible/lnInclude",
        os.getenv("FOAM_SRC")+"/thermophysicalModels/basic/lnInclude",
        os.getenv("FOAM_SRC")+"/thermophysicalModels/radiation/lnInclude",
        os.getenv("FOAM_SRC")+"/TurbulenceModels/turbulenceModels/lnInclude",
        os.getenv("FOAM_SRC")+"/TurbulenceModels/compressible/lnInclude",
        os.getenv("FOAM_SRC")+"/finiteVolume/cfdTools",
        os.getenv("FOAM_SRC")+"/finiteVolume/lnInclude",
        os.getenv("FOAM_SRC")+"/meshTools/lnInclude",
        os.getenv("FOAM_SRC")+"/sampling/lnInclude",
        # These are common for all OpenFOAM executives
        os.getenv("FOAM_SRC")+"/OpenFOAM/lnInclude",
        os.getenv("FOAM_SRC")+"/OSspecific/POSIX/lnInclude",
        os.getenv("FOAM_LIBBIN"),
        # DAFoam include
        os.getenv("PETSC_DIR")+"/include",
        petsc4py.get_include(),
        os.getenv("PETSC_DIR")+"/"+os.getenv("PETSC_ARCH")+"/include",
        "../../ofLibs/lnInclude",
        "./"
        "../../include"],
    # These are from Make/options:EXE_LIBS
    libraries=[
        "compressibleTransportModels",
        "fluidThermophysicalModels",
        "specie",
        "turbulenceModels", 
        "compressibleTurbulenceModels",
        "radiationModels",
        "finiteVolume",
        "sampling",
        "meshTools",
        "fvOptions",
        "AdjointDerivativeCompressible",
        "petsc"],
    # These are pathes of linked libraries 
    library_dirs=[
        os.getenv("FOAM_LIBBIN"),
        os.getenv("FOAM_USER_LIBBIN"),
        os.getenv("PETSC_DIR")+"/lib",
        petsc4py.get_include(),
        os.getenv("PETSC_DIR")+"/"+os.getenv("PETSC_ARCH")+"/lib"],
    # All other flags for OpenFOAM, users don't need to touch this 
    extra_compile_args=[
        "-DCompressibleFlow",
        "-std=c++11",
        "-m64",
        "-DOPENFOAM_PLUS=1812",
        "-Dlinux64",
        "-DWM_ARCH_OPTION=64",
        "-DWM_DP",
        "-DWM_LABEL_SIZE=32",
        "-Wall",
        "-Wextra",
        "-Wnon-virtual-dtor",
        "-Wno-unused-parameter",
        "-Wno-invalid-offsetof",
        "-O3",
        "-DNoRepository",
        "-ftemplate-depth-100",
        "-fPIC",
        "-c"],
    # Extra link flags for OpenFOAM, users don't need to touch this 
    extra_link_args=[
        "-Xlinker",
        "--add-needed",
        "-Xlinker",
        "--no-as-needed"]
     )]


setup(
    name = libName,
    packages = [libName],
    description = "Cython wrapper for OpenFOAM",
    long_description = "Cython wrapper for OpenFOAM",
    ext_modules = cythonize(ext,language_level=3)) # languate_level=3 means python3
