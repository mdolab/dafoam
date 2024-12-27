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

    dictionary functionSubDict =
        daOption_.getAllOptions().subDict("function").subDict(outputName_);

    // loop over all parts for this functionName
    scalar fVal = 0.0;
    forAll(functionSubDict.toc(), idxJ)
    {
        // get the subDict for this part
        word functionPart = functionSubDict.toc()[idxJ];

        // get function from daFunctionList_
        label objIndx = this->getFunctionListIndex(outputName_, functionPart, daFunctionList_);
        DAFunction& daFunction = daFunctionList_[objIndx];

        // compute the objective function
        fVal += daFunction.getFunctionValue();
    }
    output[0] = fVal;
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
