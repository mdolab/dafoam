/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

    Description:
        Child class for the max value for the given valList. The max value
        is estimated using the K-S aggregation method

\*---------------------------------------------------------------------------*/

#ifndef DATimeOpMaxKS_H
#define DATimeOpMaxKS_H

#include "DATimeOp.H"
#include "addToRunTimeSelectionTable.H"
#include "mixedFvPatchFields.H"
#include "DAGlobalVar.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
      Class DATimeOpMaxKS Declaration
\*---------------------------------------------------------------------------*/

class DATimeOpMaxKS
    : public DATimeOp
{

protected:
    /// the KS coefficient
    scalar coeffKS_;

public:
    TypeName("maxKS");
    // Constructors

    //- Construct from components
    DATimeOpMaxKS(
        const word timeOpType,
        const dictionary options);

    //- Destructor
    virtual ~DATimeOpMaxKS()
    {
    }

    /// compute the timeOp value based on valList
    virtual scalar compute(
        const scalarList& valList,
        const label iStart,
        const label iEnd);

    /// compute the scaling factor for dF/d? calculation.
    virtual scalar dFScaling(
        const scalarList& valList,
        const label iStart,
        const label iEnd,
        const label timeIdx);
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
