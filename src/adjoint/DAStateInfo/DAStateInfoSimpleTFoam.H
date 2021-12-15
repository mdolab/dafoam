/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v2

    Description:
        Child class for DASimpleTFoam

\*---------------------------------------------------------------------------*/

#ifndef DAStateInfoSimpleTFoam_H
#define DAStateInfoSimpleTFoam_H

#include "DAStateInfo.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
      Class DAStateInfoSimpleTFoam Declaration
\*---------------------------------------------------------------------------*/

class DAStateInfoSimpleTFoam
    : public DAStateInfo
{

public:
    TypeName("DASimpleTFoam");
    // Constructors

    //- Construct from components
    DAStateInfoSimpleTFoam(
        const word modelType,
        const fvMesh& mesh,
        const DAOption& daOption,
        const DAModel& daModel);

    //- Destructor
    virtual ~DAStateInfoSimpleTFoam()
    {
    }
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
