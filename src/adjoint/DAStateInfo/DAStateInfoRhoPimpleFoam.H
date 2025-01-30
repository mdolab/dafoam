/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

    Description:
        Child class for DARhoPimpleFoam

\*---------------------------------------------------------------------------*/

#ifndef DAStateInfoRhoPimpleFoam_H
#define DAStateInfoRhoPimpleFoam_H

#include "DAStateInfo.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
      Class DAStateInfoRhoPimpleFoam Declaration
\*---------------------------------------------------------------------------*/

class DAStateInfoRhoPimpleFoam
    : public DAStateInfo
{

public:
    TypeName("DARhoPimpleFoam");
    // Constructors

    //- Construct from components
    DAStateInfoRhoPimpleFoam(
        const word modelType,
        const fvMesh& mesh,
        const DAOption& daOption,
        const DAModel& daModel);

    //- Destructor
    virtual ~DAStateInfoRhoPimpleFoam()
    {
    }
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
