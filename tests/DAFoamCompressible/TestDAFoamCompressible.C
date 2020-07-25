/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v2

\*---------------------------------------------------------------------------*/
#include "TestDAFoamCompressible.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

// Constructors
TestDAFoamCompressible::TestDAFoamCompressible(char* argsAll)
    : argsAll_(argsAll)
{
}

TestDAFoamCompressible::~TestDAFoamCompressible()
{
}

label TestDAFoamCompressible::testDAStateInfo(PyObject* pyDict)
{
    autoPtr<argList> argsPtr_;
#include "setArgs.H"
#include "setRootCasePython.H"
#include "createTime.H"
#include "createMesh.H"
#include "createFields.H"

    label testErrors = 0;
/*
    DAOption daOption(mesh, pyDict);

    autoPtr<DATurbulenceModel> daTurbmodel(
        DATurbulenceModel::New(mesh));

    DAModel daModel(mesh);

    autoPtr<DARegState> daRegState(DARegState::New(mesh));

    const HashTable<wordList>& regStates = daRegState->getRegStates();

    HashTable<wordList> regStatesRef;

    regStatesRef.set("volScalarStates", {});
    regStatesRef.set("volVectorStates", {});
    regStatesRef.set("modelStates", {});
    regStatesRef.set("surfaceScalarStates", {});
    regStatesRef["volScalarStates"].append("p");
    regStatesRef["volScalarStates"].append("T");
    regStatesRef["modelStates"].append("nut");
    regStatesRef["volVectorStates"].append("U");
    regStatesRef["surfaceScalarStates"].append("phi");

    daRegState->correctModelStates(regStatesRef["modelStates"]);

    if (regStates != regStatesRef)
    {
        Pout << "compressible error in DARegState!" << endl;
        testErrors += 1;
    }
*/
    return testErrors;
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
