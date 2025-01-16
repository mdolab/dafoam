/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

    Description:
        Time operation for functions

\*---------------------------------------------------------------------------*/

#ifndef DATimeOp_H
#define DATimeOp_H

#include "runTimeSelectionTables.H"
#include "fvOptions.H"
#include "DAOption.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                    Class DATimeOp Declaration
\*---------------------------------------------------------------------------*/

class DATimeOp
{

private:
    /// Disallow default bitwise copy construct
    DATimeOp(const DATimeOp&);

    /// Disallow default bitwise assignment
    void operator=(const DATimeOp&);

protected:
    /// type of the timeOp
    const word timeOpType_;

public:
    /// Runtime type information
    TypeName("DATimeOp");

    // Declare run-time constructor selection table
    declareRunTimeSelectionTable(
        autoPtr,
        DATimeOp,
        dictionary,
        (
            const word timeOpType),
        (
            timeOpType));

    // Constructors

    //- Construct from components
    DATimeOp(const word timeOpType);

    // Selectors

    //- Return a reference to the selected model
    static autoPtr<DATimeOp> New(const word timeOpType);

    //- Destructor
    virtual ~DATimeOp()
    {
    }

    /// compute the timeOp value based on valList
    virtual scalar compute(
        const scalarList& valList,
        const label iStart,
        const label iEnd) = 0;
    
    /// compute the scaling factor for dF/d? calculation.
    virtual scalar dFScaling(
        const scalarList& valList,
        const label iStart,
        const label iEnd,
        const label timeIdx) = 0;
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
