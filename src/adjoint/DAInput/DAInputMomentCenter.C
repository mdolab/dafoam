/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v5

\*---------------------------------------------------------------------------*/

#include "DAInputMomentCenter.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(DAInputMomentCenter, 0);
addToRunTimeSelectionTable(DAInput, DAInputMomentCenter, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DAInputMomentCenter::DAInputMomentCenter(
    const word inputName,
    const word inputType,
    fvMesh& mesh,
    const DAOption& daOption,
    const DAModel& daModel,
    const DAIndex& daIndex)
    : DAInput(
          inputName,
          inputType,
          mesh,
          daOption,
          daModel,
          daIndex)
{
}

void DAInputMomentCenter::run(const scalarList& input)
{
    /*
    Description:
        Assign the input array [x, y, z] to DAGlobalVar::momentCenter
        DAFunctionMoment will read it from DAGlobalVar to update its momentCenter_
    */

#ifndef CODI_ADR
    Info << "DAInputMomentCenter. " << endl;
    Info << "Setting momentCenter = " << input << endl;
#endif

    DAGlobalVar& globalVar =
        const_cast<DAGlobalVar&>(mesh_.thisDb().lookupObject<DAGlobalVar>("DAGlobalVar"));
    globalVar.momentCenter[0] = input[0];
    globalVar.momentCenter[1] = input[1];
    globalVar.momentCenter[2] = input[2];
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
