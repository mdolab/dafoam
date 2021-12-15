/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v2

    Description:
        Child class for DAScalarTransportFoam

\*---------------------------------------------------------------------------*/

#ifndef DAStateInfoScalarTransportFoam_H
#define DAStateInfoScalarTransportFoam_H

#include "DAStateInfo.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
      Class DAStateInfoScalarTransportFoam Declaration
\*---------------------------------------------------------------------------*/

class DAStateInfoScalarTransportFoam
    : public DAStateInfo
{

public:
    TypeName("DAScalarTransportFoam");
    // Constructors

    //- Construct from components
    DAStateInfoScalarTransportFoam(
        const word modelType,
        const fvMesh& mesh,
        const DAOption& daOption,
        const DAModel& daModel);

    //- Destructor
    virtual ~DAStateInfoScalarTransportFoam()
    {
    }
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
