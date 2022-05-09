/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v3

    Description:
        Child class for center of pressure

\*---------------------------------------------------------------------------*/

#ifndef DAFPAdjSimpleFoam_H
#define DAFPAdjSimpleFoam_H

#include "DAFPAdj.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
      Class DAFPAdjSimpleFoam Declaration
\*---------------------------------------------------------------------------*/

class DAFPAdjSimpleFoam
    : public DAFPAdj
{

public:
    TypeName("DASimpleFoam");
    // Constructors

    //- Construct from components
    DAFPAdjSimpleFoam(
        const word modelType,
        const fvMesh& mesh,
        const DAOption& daOption,
        const DAModel& daModel,
        const DAIndex& daIndex);

    //- Destructor
    virtual ~DAFPAdjSimpleFoam()
    {
    }

    /// solve the adjoint using the fixed-point iteration method
    virtual label run(
        Vec dFdW,
        Vec psi);
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
