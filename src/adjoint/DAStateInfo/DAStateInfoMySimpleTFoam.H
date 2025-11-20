/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v3

    Description:
        Child class for DASimpleTFoam

\*---------------------------------------------------------------------------*/

#ifndef DAStateInfoMySimpleTFoam_H
#define DAStateInfoMySimpleTFoam_H

#include "DAStateInfo.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
      Class DAStateInfoMySimpleTFoam Declaration
\*---------------------------------------------------------------------------*/

class DAStateInfoMySimpleTFoam
    : public DAStateInfo
{

public:
    TypeName("DAMySimpleTFoam");
    // Constructors

    //- Construct from components
    DAStateInfoMySimpleTFoam(
        const word modelType,
        const fvMesh& mesh,
        const DAOption& daOption,
        const DAModel& daModel);

    //- Destructor
    virtual ~DAStateInfoMySimpleTFoam()
    {
    }
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
