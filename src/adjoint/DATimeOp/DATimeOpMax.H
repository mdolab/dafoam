/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

    Description:
        Child class for the max value for the given valList. The max value
        is estimated using the K-S aggregation method

\*---------------------------------------------------------------------------*/

#ifndef DATimeOpMax_H
#define DATimeOpMax_H

#include "DATimeOp.H"
#include "addToRunTimeSelectionTable.H"
#include "mixedFvPatchFields.H"
#include "DAGlobalVar.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
      Class DATimeOpMax Declaration
\*---------------------------------------------------------------------------*/

class DATimeOpMax
    : public DATimeOp
{

protected:
    /// the KS coefficient
    scalar coeffKS_;

    /// max value mode
    word mode_ = "KS";

public:
    TypeName("max");
    // Constructors

    //- Construct from components
    DATimeOpMax(
        const word timeOpType,
        const dictionary options);

    //- Destructor
    virtual ~DATimeOpMax()
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
