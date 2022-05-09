/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v3

    Description:
        Child class for DAPisoFoam

\*---------------------------------------------------------------------------*/

#ifndef DAStateInfoPisoFoam_H
#define DAStateInfoPisoFoam_H

#include "DAStateInfo.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
      Class DAStateInfoPisoFoam Declaration
\*---------------------------------------------------------------------------*/

class DAStateInfoPisoFoam
    : public DAStateInfo
{

public:
    TypeName("DAPisoFoam");
    // Constructors

    //- Construct from components
    DAStateInfoPisoFoam(
        const word modelType,
        const fvMesh& mesh,
        const DAOption& daOption,
        const DAModel& daModel);

    //- Destructor
    virtual ~DAStateInfoPisoFoam()
    {
    }
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
