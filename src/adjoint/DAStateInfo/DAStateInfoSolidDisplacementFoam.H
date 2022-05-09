/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v3

    Description:
        Child class for DASolidDisplacementFoam

\*---------------------------------------------------------------------------*/

#ifndef DAStateInfoSolidDisplacementFoam_H
#define DAStateInfoSolidDisplacementFoam_H

#include "DAStateInfo.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
      Class DAStateInfoSolidDisplacementFoam Declaration
\*---------------------------------------------------------------------------*/

class DAStateInfoSolidDisplacementFoam
    : public DAStateInfo
{

public:
    TypeName("DASolidDisplacementFoam");
    // Constructors

    //- Construct from components
    DAStateInfoSolidDisplacementFoam(
        const word modelType,
        const fvMesh& mesh,
        const DAOption& daOption,
        const DAModel& daModel);

    //- Destructor
    virtual ~DAStateInfoSolidDisplacementFoam()
    {
    }
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
