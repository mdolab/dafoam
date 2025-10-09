/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

\*---------------------------------------------------------------------------*/

#include "DATimeOpMax.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(DATimeOpMax, 0);
addToRunTimeSelectionTable(DATimeOp, DATimeOpMax, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DATimeOpMax::DATimeOpMax(
    const word timeOpType,
    const dictionary options)
    : DATimeOp(timeOpType, options)
{
    mode_ = options.lookupOrDefault<word>("timeOpMaxMode", "KS");
    if (mode_ == "KS")
    {
        coeffKS_ = options.getScalar("timeOpMaxKSCoeff");
    }
}

scalar DATimeOpMax::compute(
    const scalarList& valList,
    const label iStart,
    const label iEnd)
{
    // return the estimated max value from valList
    // KS = log( sum( exp(x_i*c) ) )/c

    if (mode_ == "KS")
    {
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
    else if (mode_ == "orig")
    {
        scalar maxVal = -1e16;
        for (label i = iStart; i <= iEnd; i++)
        {
            if (valList[i] > maxVal)
            {
                maxVal = valList[i];
            }
        }
        return maxVal;
    }
    else
    {
        FatalErrorIn(" ") << "mode not supported! Options are orig or KS " << abort(FatalError);
        return -1e16;
    }
}

scalar DATimeOpMax::dFScaling(
    const scalarList& valList,
    const label iStart,
    const label iEnd,
    const label timeIdx)
{
    // F = log( sum( exp(f_i*c) ) )/c
    // dF/dx = 1/(sum(exp(f_i*c))) * exp(f_i*c) * df_i/dx
    // so the scaling is 1/(sum(exp(f_i*c))) * exp(f_i*c)
    // NOTE: the scaling depends on timeIdx!

    if (mode_ != "KS")
    {
        FatalErrorIn(" ") << "mode orig is selected! You should not run the adjoint! " << abort(FatalError);
    }

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
