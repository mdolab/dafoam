/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

    Description:
        Child class for the average value for the given valList

\*---------------------------------------------------------------------------*/

#ifndef DATimeOpAverage_H
#define DATimeOpAverage_H

#include "DATimeOp.H"
#include "addToRunTimeSelectionTable.H"
#include "mixedFvPatchFields.H"
#include "DAGlobalVar.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
      Class DATimeOpAverage Declaration
\*---------------------------------------------------------------------------*/

class DATimeOpAverage
    : public DATimeOp
{

protected:
public:
    TypeName("average");
    // Constructors

    //- Construct from components
    DATimeOpAverage(
        const word timeOpType,
        const dictionary options);

    //- Destructor
    virtual ~DATimeOpAverage()
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
