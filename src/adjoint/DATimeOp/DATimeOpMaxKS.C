/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

\*---------------------------------------------------------------------------*/

#include "DATimeOpMaxKS.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(DATimeOpMaxKS, 0);
addToRunTimeSelectionTable(DATimeOp, DATimeOpMaxKS, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DATimeOpMaxKS::DATimeOpMaxKS(
    const word timeOpType,
    const dictionary options)
    : DATimeOp(timeOpType, options)
{
    coeffKS_ = options.getScalar("coeffKS");
}

scalar DATimeOpMaxKS::compute(
    const scalarList& valList,
    const label iStart,
    const label iEnd)
{
    // return the estimated max value from valList
    // KS = log( sum( exp(x_i*c) ) )/c
    scalar maxKS = 0.0;
    // NOTE. We need to use <= here
    for (label i = iStart; i <= iEnd; i++)
    {
        maxKS += exp(coeffKS_ * valList[i]);
    }
    if (maxKS > 1e200)
    {
        FatalErrorIn(" ") << "KS function summation term too large! "
                          << "Reduce coeffKS! " << abort(FatalError);
    }
    maxKS = log(maxKS) / coeffKS_;
    return maxKS;
}

scalar DATimeOpMaxKS::dFScaling(
    const scalarList& valList,
    const label iStart,
    const label iEnd,
    const label timeIdx)
{
    // F = log( sum( exp(f_i*c) ) )/c
    // dF/dx = 1/(sum(exp(f_i*c))) * exp(f_i*c) * df_i/dx
    // so the scaling is 1/(sum(exp(f_i*c))) * exp(f_i*c)
    // NOTE: the scaling depends on timeIdx!

    scalar sum = 0.0;
    for (label i = iStart; i <= iEnd; i++)
    {
        sum += exp(coeffKS_ * valList[i]);
    }
    scalar scaling = 1.0 / sum * exp(valList[timeIdx] * coeffKS_);

    return scaling;
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
