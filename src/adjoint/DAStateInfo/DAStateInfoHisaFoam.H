/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v5

    Description:
        Child class for DAHisaFoam

\*---------------------------------------------------------------------------*/

#ifndef DAStateInfoHisaFoam_H
#define DAStateInfoHisaFoam_H

#include "DAStateInfo.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
      Class DAStateInfoHisaFoam Declaration
\*---------------------------------------------------------------------------*/

class DAStateInfoHisaFoam
    : public DAStateInfo
{

public:
    TypeName("DAHisaFoam");
    // Constructors

    //- Construct from components
    DAStateInfoHisaFoam(
        const word modelType,
        const fvMesh& mesh,
        const DAOption& daOption,
        const DAModel& daModel);

    //- Destructor
    virtual ~DAStateInfoHisaFoam()
    {
    }
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
