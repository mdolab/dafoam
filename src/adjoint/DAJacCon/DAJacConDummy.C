/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v3

\*---------------------------------------------------------------------------*/

#include "DAJacConDummy.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(DAJacConDummy, 0);
addToRunTimeSelectionTable(DAJacCon, DAJacConDummy, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DAJacConDummy::DAJacConDummy(
    const word modelType,
    const fvMesh& mesh,
    const DAOption& daOption,
    const DAModel& daModel,
    const DAIndex& daIndex)
    : DAJacCon(modelType, mesh, daOption, daModel, daIndex)
{
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

void DAJacConDummy::setupJacCon(const dictionary& options)
{
    // no need to do anything
}

void DAJacConDummy::initializeJacCon(const dictionary& options)
{
    // no need to do anything
}

void DAJacConDummy::clear()
{
    /*
    Description:
        Clear all members to avoid memory leak because we will initalize 
        multiple objects of DAJacCon. Here we need to delete all members
        in the parent and child classes
    */
    
    // no need to do anything
}

} // End namespace Foam

// ************************************************************************* //
