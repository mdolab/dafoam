/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v3

\*---------------------------------------------------------------------------*/

#include "DAResidualDummy.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(DAResidualDummy, 0);
addToRunTimeSelectionTable(DAResidual, DAResidualDummy, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DAResidualDummy::DAResidualDummy(
    const word modelType,
    const fvMesh& mesh,
    const DAOption& daOption,
    const DAModel& daModel,
    const DAIndex& daIndex)
    : DAResidual(modelType, mesh, daOption, daModel, daIndex)
{

}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

void DAResidualDummy::clear()
{
}

void DAResidualDummy::calcResiduals(const dictionary& options)
{
}

void DAResidualDummy::updateIntermediateVariables()
{

}

void DAResidualDummy::correctBoundaryConditions()
{

}

} // End namespace Foam

// ************************************************************************* //
