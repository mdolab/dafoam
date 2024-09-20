/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

    Description:
        Child class for DAPimpleFoam

\*---------------------------------------------------------------------------*/

#ifndef DAStateInfoPimpleFoam_H
#define DAStateInfoPimpleFoam_H

#include "DAStateInfo.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
      Class DAStateInfoPimpleFoam Declaration
\*---------------------------------------------------------------------------*/

class DAStateInfoPimpleFoam
    : public DAStateInfo
{

public:
    TypeName("DAPimpleFoam");
    // Constructors

    //- Construct from components
    DAStateInfoPimpleFoam(
        const word modelType,
        const fvMesh& mesh,
        const DAOption& daOption,
        const DAModel& daModel);

    //- Destructor
    virtual ~DAStateInfoPimpleFoam()
    {
    }
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
