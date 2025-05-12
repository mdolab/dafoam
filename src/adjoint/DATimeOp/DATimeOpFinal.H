/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

    Description:
        Child class for the final function value for the given valList

\*---------------------------------------------------------------------------*/

#ifndef DATimeOpFinal_H
#define DATimeOpFinal_H

#include "DATimeOp.H"
#include "addToRunTimeSelectionTable.H"
#include "mixedFvPatchFields.H"
#include "DAGlobalVar.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
      Class DATimeOpFinal Declaration
\*---------------------------------------------------------------------------*/

class DATimeOpFinal
    : public DATimeOp
{

protected:
public:
    TypeName("final");
    // Constructors

    //- Construct from components
    DATimeOpFinal(
        const word timeOpType,
        const dictionary options);

    //- Destructor
    virtual ~DATimeOpFinal()
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
