/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

\*---------------------------------------------------------------------------*/

#include "DAOutputFunction.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(DAOutputFunction, 0);
addToRunTimeSelectionTable(DAOutput, DAOutputFunction, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DAOutputFunction::DAOutputFunction(
    const word outputName,
    const word outputType,
    fvMesh& mesh,
    const DAOption& daOption,
    DAModel& daModel,
    const DAIndex& daIndex,
    DAResidual& daResidual,
    UPtrList<DAFunction>& daFunctionList)
    : DAOutput(
        outputName,
        outputType,
        mesh,
        daOption,
        daModel,
        daIndex,
        daResidual,
        daFunctionList)
{
}

void DAOutputFunction::run(scalarList& output)
{
    /*
    Description:
        Compute the function value and assign them to the output array
    */

    word functionName = outputName_;

    label idxI = this->getFunctionListIndex(functionName, daFunctionList_);
    DAFunction& daFunction = daFunctionList_[idxI];

    // compute the objective function
    scalar fVal = daFunction.calcFunction();

    output[0] = fVal;
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
