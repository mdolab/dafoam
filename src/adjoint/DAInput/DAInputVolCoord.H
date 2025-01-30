/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

    Description:
        Child class for volume coordinate input

\*---------------------------------------------------------------------------*/

#ifndef DAInputVolCoord_H
#define DAInputVolCoord_H

#include "DAInput.H"
#include "addToRunTimeSelectionTable.H"
#include "mixedFvPatchFields.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
      Class DAInputVolCoord Declaration
\*---------------------------------------------------------------------------*/

class DAInputVolCoord
    : public DAInput
{

protected:
public:
    TypeName("volCoord");
    // Constructors

    //- Construct from components
    DAInputVolCoord(
        const word inputName,
        const word inputType,
        fvMesh& mesh,
        const DAOption& daOption,
        const DAModel& daModel,
        const DAIndex& daIndex);

    //- Destructor
    virtual ~DAInputVolCoord()
    {
    }

    virtual void run(const scalarList& input);

    virtual label size()
    {
        return daIndex_.nLocalXv;
    }

    virtual label distributed()
    {
        return 1;
    }
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
