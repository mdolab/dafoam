/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v5

    Description:
        Child class for DATopoChtFoam

\*---------------------------------------------------------------------------*/

#ifndef DAStateInfoTopoChtFoam_H
#define DAStateInfoTopoChtFoam_H

#include "DAStateInfo.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
      Class DAStateInfoTopoChtFoam Declaration
\*---------------------------------------------------------------------------*/

class DAStateInfoTopoChtFoam
    : public DAStateInfo
{

public:
    TypeName("DATopoChtFoam");
    // Constructors

    //- Construct from components
    DAStateInfoTopoChtFoam(
        const word modelType,
        const fvMesh& mesh,
        const DAOption& daOption,
        const DAModel& daModel);

    //- Destructor
    virtual ~DAStateInfoTopoChtFoam()
    {
    }
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
