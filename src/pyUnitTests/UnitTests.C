/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

\*---------------------------------------------------------------------------*/
#include "UnitTests.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
// initialize the static variable, which will be used in forward mode AD
// computation for AOA and BC derivatives
scalar Foam::DAUtility::angleOfAttackRadForwardAD = -9999.0;

// initialize the python call back function static pointers
void* Foam::DAUtility::pyCalcBeta = NULL;
pyComputeInterface Foam::DAUtility::pyCalcBetaInterface = NULL;

void* Foam::DAUtility::pyCalcBetaJacVecProd = NULL;
pyJacVecProdInterface Foam::DAUtility::pyCalcBetaJacVecProdInterface = NULL;

void* Foam::DAUtility::pySetModelName = NULL;
pySetCharInterface Foam::DAUtility::pySetModelNameInterface = NULL;

scalar Foam::DAUtility::primalMaxInitRes_ = -1e16;

namespace Foam
{

// Constructors
UnitTests::UnitTests()
{
}

UnitTests::~UnitTests()
{
}

void UnitTests::runDAOptionTest1(
    char* argsAll,
    PyObject* pyOptions)
{
    Info << "Run the tests" << endl;
    dictionary options;
    DAUtility::pyDict2OFDict(pyOptions, options);
    Info << options << endl;
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
