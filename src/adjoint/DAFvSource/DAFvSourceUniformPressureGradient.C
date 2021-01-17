/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v2

\*---------------------------------------------------------------------------*/

#include "DAFvSourceUniformPressureGradient.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(DAFvSourceUniformPressureGradient, 0);
addToRunTimeSelectionTable(DAFvSource, DAFvSourceUniformPressureGradient, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DAFvSourceUniformPressureGradient::DAFvSourceUniformPressureGradient(
    const word modelType,
    const fvMesh& mesh,
    const DAOption& daOption,
    const DAModel& daModel,
    const DAIndex& daIndex)
    : DAFvSource(modelType, mesh, daOption, daModel, daIndex)
{
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

void DAFvSourceUniformPressureGradient::calcFvSource(volVectorField& fvSource)
{
    /*
    Description:
        Just add a uniform pressure gradient to fvSource

    Example:
        An example of the fvSource in pyOptions in pyDAFoam can be
        defOptions = 
        {
            "fvSource"
            {
                "gradP"
                {
                    "type": "uniformPressureGradient",
                    "value": 1e-3,
                    "direction": [1.0, 0.0, 0.0],
                },
            }
        }
    */

    forAll(fvSource, idxI)
    {
        fvSource[idxI] = vector::zero;
    }

    const dictionary& allOptions = daOption_.getAllOptions();

    dictionary fvSourceSubDict = allOptions.subDict("fvSource");

    word gradPName = fvSourceSubDict.toc()[0];
    scalar gradPValue = fvSourceSubDict.subDict(gradPName).getScalar("value");
    scalarList direction;
    fvSourceSubDict.subDict(gradPName).readEntry<scalarList>("direction", direction);

    vector directionVec = {direction[0], direction[1], direction[2]};

    forAll(fvSource, idxI)
    {
        fvSource[idxI] = gradPValue * directionVec;
    }

    fvSource.correctBoundaryConditions();
}

} // End namespace Foam

// ************************************************************************* //
