/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

    Description:
        Child class for DARhoSimpleCFoam

\*---------------------------------------------------------------------------*/

#ifndef DAStateInfoRhoSimpleCFoam_H
#define DAStateInfoRhoSimpleCFoam_H

#include "DAStateInfo.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
      Class DAStateInfoRhoSimpleCFoam Declaration
\*---------------------------------------------------------------------------*/

class DAStateInfoRhoSimpleCFoam
    : public DAStateInfo
{

public:
    TypeName("DARhoSimpleCFoam");
    // Constructors

    //- Construct from components
    DAStateInfoRhoSimpleCFoam(
        const word modelType,
        const fvMesh& mesh,
        const DAOption& daOption,
        const DAModel& daModel);

    //- Destructor
    virtual ~DAStateInfoRhoSimpleCFoam()
    {
    }
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
