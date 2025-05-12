/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

\*---------------------------------------------------------------------------*/

#include "DATimeOpFinal.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(DATimeOpFinal, 0);
addToRunTimeSelectionTable(DATimeOp, DATimeOpFinal, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DATimeOpFinal::DATimeOpFinal(
    const word timeOpType,
    const dictionary options)
    : DATimeOp(timeOpType, options)
{
}

scalar DATimeOpFinal::compute(
    const scalarList& valList,
    const label iStart,
    const label iEnd)
{
    // just return the last value from valList
    return valList[iEnd];
}

scalar DATimeOpFinal::dFScaling(
    const scalarList& valList,
    const label iStart,
    const label iEnd,
    const label timeIdx)
{
    // the dFScaling is alway 1
    return 1.0;
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
