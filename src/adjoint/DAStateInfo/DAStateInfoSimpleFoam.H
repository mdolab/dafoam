/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

    Description:
        Child class for DASimpleFoam

\*---------------------------------------------------------------------------*/

#ifndef DAStateInfoSimpleFoam_H
#define DAStateInfoSimpleFoam_H

#include "DAStateInfo.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
      Class DAStateInfoSimpleFoam Declaration
\*---------------------------------------------------------------------------*/

class DAStateInfoSimpleFoam
    : public DAStateInfo
{

public:
    TypeName("DASimpleFoam");
    // Constructors

    //- Construct from components
    DAStateInfoSimpleFoam(
        const word modelType,
        const fvMesh& mesh,
        const DAOption& daOption,
        const DAModel& daModel);

    //- Destructor
    virtual ~DAStateInfoSimpleFoam()
    {
    }
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
