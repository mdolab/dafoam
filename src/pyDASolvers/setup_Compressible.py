"""
    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v2

    Description:
        Cython setup file for wrapping OpenFOAM libraries and solvers.
        One needs to set include dirs/files and flags according to the
        information in Make/options and Make/files in OpenFOAM libraries
        and solvers. Then, follow the detailed instructions below. The
        python naming convention is to add "py" before the C++ class name
"""

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import os
import petsc4py

os.environ["CC"] = "mpicc"
os.environ["CXX"] = "mpicxx"

solverName = "pyDASolverCompressible"

if os.getenv("WM_CODI_AD_MODE") is None:
    libSuffix = ""
    codiADMode = "CODI_AD_NONE"
elif os.getenv("WM_CODI_AD_MODE") == "CODI_AD_FORWARD":
    libSuffix = "ADF"
    codiADMode = os.getenv("WM_CODI_AD_MODE")
elif os.getenv("WM_CODI_AD_MODE") == "CODI_AD_REVERSE":
    libSuffix = "ADR"
    codiADMode = os.getenv("WM_CODI_AD_MODE")

# These setup should reproduce calling wmake to compile OpenFOAM libraries and solvers
ext = [
    Extension(
        solverName + libSuffix,
        # All source files, taken from Make/files
        sources=["pyDASolvers.pyx", "DASolvers.C"],
        # All include dirs, refer to Make/options in OpenFOAM
        include_dirs=[
            # These are from Make/options:EXE_INC
            os.getenv("FOAM_SRC") + "/transportModels/compressible/lnInclude",
            os.getenv("FOAM_SRC") + "/thermophysicalModels/basic/lnInclude",
            os.getenv("FOAM_SRC") + "/TurbulenceModels/turbulenceModels/lnInclude",
            os.getenv("FOAM_SRC") + "/TurbulenceModels/compressible/lnInclude",
            os.getenv("FOAM_SRC") + "/finiteVolume/cfdTools",
            os.getenv("FOAM_SRC") + "/finiteVolume/lnInclude",
            os.getenv("FOAM_SRC") + "/meshTools/lnInclude",
            os.getenv("FOAM_SRC") + "/sampling/lnInclude",
            os.getenv("FOAM_SRC") + "/fileFormats/lnInclude",
            os.getenv("FOAM_SRC") + "/surfMesh/lnInclude",
            # These are common for all OpenFOAM executives
            os.getenv("FOAM_SRC") + "/OpenFOAM/lnInclude",
            os.getenv("FOAM_SRC") + "/OSspecific/POSIX/lnInclude",
            os.getenv("FOAM_LIBBIN"),
            # CoDiPack and MeDiPack
            os.getenv("FOAM_SRC") + "/codipack/include",
            os.getenv("FOAM_SRC") + "/medipack/include",
            os.getenv("FOAM_SRC") + "/medipack/src",
            # DAFoam include
            os.getenv("PETSC_DIR") + "/include",
            petsc4py.get_include(),
            os.getenv("PETSC_DIR") + "/" + os.getenv("PETSC_ARCH") + "/include",
            "../adjoint/lnInclude",
            "../include",
            "./",
        ],
        # These are from Make/options:EXE_LIBS
        libraries=[
            "compressibleTransportModels" + libSuffix,
            "fluidThermophysicalModels" + libSuffix,
            "specie" + libSuffix,
            "turbulenceModels" + libSuffix,
            "compressibleTurbulenceModels" + libSuffix,
            "finiteVolume" + libSuffix,
            "sampling" + libSuffix,
            "meshTools" + libSuffix,
            "fvOptions" + libSuffix,
            "DAFoamCompressible" + libSuffix,
            "petsc",
        ],
        # These are pathes of linked libraries
        library_dirs=[
            os.getenv("FOAM_LIBBIN"),
            os.getenv("DAFOAM_ROOT_PATH") + "/OpenFOAM/sharedLibs",
            os.getenv("PETSC_DIR") + "/lib",
            petsc4py.get_include(),
            os.getenv("PETSC_DIR") + "/" + os.getenv("PETSC_ARCH") + "/lib",
        ],
        # All other flags for OpenFOAM, users don't need to touch this
        extra_compile_args=[
            "-std=c++11",
            "-DCompressibleFlow",
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
            "-c",
            "-D" + codiADMode,
        ],
        # Extra link flags for OpenFOAM, users don't need to touch this
        extra_link_args=["-Xlinker", "--add-needed", "-Xlinker", "--no-as-needed"],
    )
]


setup(
    name=solverName + libSuffix,
    packages=[solverName + libSuffix],
    description="Cython wrapper for OpenFOAM",
    long_description="Cython wrapper for OpenFOAM",
    ext_modules=cythonize(ext, language_level=3),
)  # languate_level=3 means python3
