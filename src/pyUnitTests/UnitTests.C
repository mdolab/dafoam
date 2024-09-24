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

void UnitTests::runDAUtilityTest1(
    char* argsAll_,
    PyObject* pyOptions)
{
#include "setArgs.H"
#include "setRootCase.H"
#include "createTime.H"
#include "createMesh.H"
    Info << "runDAUtilityTest1" << endl;
    dictionary ofOptions;
    DAUtility::pyDict2OFDict(pyOptions, ofOptions);
    
    IOdictionary ofOptionsIO(
        IOobject(
            "test_dict",
            "misc_files",
            mesh,
            IOobject::MUST_READ,
            IOobject::NO_WRITE));

    if (ofOptions != ofOptionsIO.subDict("ofOptions"))
    {
        Info << "********* pyDict2OFDict test failed! **********" << endl;
        Info << "ofOptions" << ofOptions << endl;
        Info << "ofOptionsIO" << ofOptionsIO.subDict("ofOptions") << endl;
    }
    else
    {
        Info << "pyDict2OFDict test passed!" << endl;
    }

    Info << "runDAUtilityTest1 Passed!" << endl;
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
