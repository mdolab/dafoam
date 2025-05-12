/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

\*---------------------------------------------------------------------------*/

#include "DATimeOpAverage.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(DATimeOpAverage, 0);
addToRunTimeSelectionTable(DATimeOp, DATimeOpAverage, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DATimeOpAverage::DATimeOpAverage(
    const word timeOpType,
    const dictionary options)
    : DATimeOp(timeOpType, options)
{
}

scalar DATimeOpAverage::compute(
    const scalarList& valList,
    const label iStart,
    const label iEnd)
{
    // return the average value from valList
    scalar avg = 0.0;
    // NOTE. We need to use <= here
    for (label i = iStart; i <= iEnd; i++)
    {
        avg += valList[i];
    }
    avg /= (iEnd - iStart + 1);
    return avg;
}

scalar DATimeOpAverage::dFScaling(
    const scalarList& valList,
    const label iStart,
    const label iEnd,
    const label timeIdx)
{
    // return 1/N as the dF scaling

    scalar scaling = 1.0 / (iEnd - iStart + 1);

    return scaling;
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
