/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v2

    Description:
        Child class for DARhoSimpleFoam

\*---------------------------------------------------------------------------*/

#ifndef DAStateInfoRhoSimpleFoam_H
#define DAStateInfoRhoSimpleFoam_H

#include "DAStateInfo.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
      Class DAStateInfoRhoSimpleFoam Declaration
\*---------------------------------------------------------------------------*/

class DAStateInfoRhoSimpleFoam
    : public DAStateInfo
{

public:
    TypeName("DARhoSimpleFoam");
    // Constructors

    //- Construct from components
    DAStateInfoRhoSimpleFoam(
        const word modelType,
        const fvMesh& mesh,
        const DAOption& daOption,
        const DAModel& daModel);

    //- Destructor
    virtual ~DAStateInfoRhoSimpleFoam()
    {
    }
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
