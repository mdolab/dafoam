/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

    Description:
        Child class for DAHeatTransferFoam

\*---------------------------------------------------------------------------*/

#ifndef DAStateInfoHeatTransferFoam_H
#define DAStateInfoHeatTransferFoam_H

#include "DAStateInfo.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
      Class DAStateInfoHeatTransferFoam Declaration
\*---------------------------------------------------------------------------*/

class DAStateInfoHeatTransferFoam
    : public DAStateInfo
{

public:
    TypeName("DAHeatTransferFoam");
    // Constructors

    //- Construct from components
    DAStateInfoHeatTransferFoam(
        const word modelType,
        const fvMesh& mesh,
        const DAOption& daOption,
        const DAModel& daModel);

    //- Destructor
    virtual ~DAStateInfoHeatTransferFoam()
    {
    }
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
