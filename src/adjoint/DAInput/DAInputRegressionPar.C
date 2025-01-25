/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

\*---------------------------------------------------------------------------*/

#include "DAInputRegressionPar.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(DAInputRegressionPar, 0);
addToRunTimeSelectionTable(DAInput, DAInputRegressionPar, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DAInputRegressionPar::DAInputRegressionPar(
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
        daIndex),
      daRegression_(const_cast<DARegression&>(
          mesh.thisDb().lookupObject<DARegression>("DARegression")))
{
}

void DAInputRegressionPar::run(const scalarList& input)
{
    /*
    Description:
        Assign the input to OF fields
    */

    label nParameters = this->size();
    for (label i = 0; i < nParameters; i++)
    {
        daRegression_.setParameter(inputName_, i, input[i]);
    }

    daRegression_.compute();
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
