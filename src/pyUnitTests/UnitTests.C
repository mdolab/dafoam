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
#include "setRootCasePython.H"
#include "createTime.H"
#include "createMesh.H"

    Info << "runDAUtilityTest1" << endl;

    // test pyDict2OFDict
    dictionary ofOptions;
    DAUtility::pyDict2OFDict(pyOptions, ofOptions);

    IOdictionary ofOptionsIO(
        IOobject(
            "test_dict",
            mesh.time().constant(),
            mesh,
            IOobject::MUST_READ,
            IOobject::NO_WRITE));

    if (ofOptions != ofOptionsIO.subDict("ofOptions"))
    {
        Info << "********* pyDict2OFDict test failed! **********" << endl;
        Info << "ofOptions" << ofOptions << endl;
        Info << "ofOptionsIO" << ofOptionsIO.subDict("ofOptions") << endl;
    }

    // test write/readMatrix
    Mat tmpMat;
    MatCreate(PETSC_COMM_WORLD, &tmpMat);
    MatSetSizes(
        tmpMat,
        PETSC_DECIDE,
        PETSC_DECIDE,
        11,
        13);
    MatSetFromOptions(tmpMat);
    MatSetUp(tmpMat);
    MatZeroEntries(tmpMat);
    PetscScalar val = Pstream::myProcNo() * Pstream::myProcNo();
    MatSetValue(tmpMat, Pstream::myProcNo(), Pstream::myProcNo(), val, INSERT_VALUES);
    MatAssemblyBegin(tmpMat, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(tmpMat, MAT_FINAL_ASSEMBLY);

    DAUtility::writeMatrixBinary(tmpMat, "tmpMat");
    DAUtility::writeMatrixASCII(tmpMat, "tmpMat");

    Mat tmpMat1;
    MatCreate(PETSC_COMM_WORLD, &tmpMat1);
    DAUtility::readMatrixBinary(tmpMat1, "tmpMat");

    PetscBool equalFlag;
    MatEqual(tmpMat, tmpMat1, &equalFlag);
    if (!equalFlag)
    {
        Info << "********* read/writeMatrixBinary test failed! **********" << endl;
    }

    // test write/readVector
    Vec tmpVec;
    VecCreate(PETSC_COMM_WORLD, &tmpVec);
    VecSetSizes(tmpVec, 17, PETSC_DETERMINE);
    VecSetFromOptions(tmpVec);
    VecSet(tmpVec, 1.0);

    DAUtility::writeVectorBinary(tmpVec, "tmpVec");
    DAUtility::writeVectorASCII(tmpVec, "tmpVec");

    Vec tmpVec1;
    VecCreate(PETSC_COMM_WORLD, &tmpVec1);
    DAUtility::readVectorBinary(tmpVec1, "tmpVec");

    VecEqual(tmpVec, tmpVec1, &equalFlag);
    if (!equalFlag)
    {
        Info << "********* read/writeVectorBinary test failed! **********" << endl;
    }

    Info << "runDAUtilityTest1 Passed!" << endl;
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
