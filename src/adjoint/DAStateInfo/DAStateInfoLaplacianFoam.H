/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v2

    Description:
        Child class for DALaplacianFoam

\*---------------------------------------------------------------------------*/

#ifndef DAStateInfoLaplacianFoam_H
#define DAStateInfoLaplacianFoam_H

#include "DAStateInfo.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
      Class DAStateInfoLaplacianFoam Declaration
\*---------------------------------------------------------------------------*/

class DAStateInfoLaplacianFoam
    : public DAStateInfo
{

public:
    TypeName("DALaplacianFoam");
    // Constructors

    //- Construct from components
    DAStateInfoLaplacianFoam(
        const word modelType,
        const fvMesh& mesh,
        const DAOption& daOption,
        const DAModel& daModel);

    //- Destructor
    virtual ~DAStateInfoLaplacianFoam()
    {
    }
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
