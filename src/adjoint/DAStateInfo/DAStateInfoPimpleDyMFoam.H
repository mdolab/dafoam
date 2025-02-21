/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

    Description:
        Child class for DAPimpleDyMFoam

\*---------------------------------------------------------------------------*/

#ifndef DAStateInfoPimpleDyMFoam_H
#define DAStateInfoPimpleDyMFoam_H

#include "DAStateInfo.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
      Class DAStateInfoPimpleDyMFoam Declaration
\*---------------------------------------------------------------------------*/

class DAStateInfoPimpleDyMFoam
    : public DAStateInfo
{

public:
    TypeName("DAPimpleDyMFoam");
    // Constructors

    //- Construct from components
    DAStateInfoPimpleDyMFoam(
        const word modelType,
        const fvMesh& mesh,
        const DAOption& daOption,
        const DAModel& daModel);

    //- Destructor
    virtual ~DAStateInfoPimpleDyMFoam()
    {
    }
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
