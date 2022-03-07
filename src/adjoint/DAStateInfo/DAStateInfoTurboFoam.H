/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v3

    Description:
        Child class for DATurboFoam

\*---------------------------------------------------------------------------*/

#ifndef DAStateInfoTurboFoam_H
#define DAStateInfoTurboFoam_H

#include "DAStateInfo.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
      Class DAStateInfoTurboFoam Declaration
\*---------------------------------------------------------------------------*/

class DAStateInfoTurboFoam
    : public DAStateInfo
{

public:
    TypeName("DATurboFoam");
    // Constructors

    //- Construct from components
    DAStateInfoTurboFoam(
        const word modelType,
        const fvMesh& mesh,
        const DAOption& daOption,
        const DAModel& daModel);

    //- Destructor
    virtual ~DAStateInfoTurboFoam()
    {
    }
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
