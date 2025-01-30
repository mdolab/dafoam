/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

    Description:
        Child class for velocity magnitude and angle of attach at a given patch

\*---------------------------------------------------------------------------*/

#ifndef DAInputPatchVelocity_H
#define DAInputPatchVelocity_H

#include "DAInput.H"
#include "addToRunTimeSelectionTable.H"
#include "mixedFvPatchFields.H"
#include "DAGlobalVar.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
      Class DAInputPatchVelocity Declaration
\*---------------------------------------------------------------------------*/

class DAInputPatchVelocity
    : public DAInput
{

protected:
public:
    TypeName("patchVelocity");
    // Constructors

    //- Construct from components
    DAInputPatchVelocity(
        const word inputName,
        const word inputType,
        fvMesh& mesh,
        const DAOption& daOption,
        const DAModel& daModel,
        const DAIndex& daIndex);

    //- Destructor
    virtual ~DAInputPatchVelocity()
    {
    }

    virtual void run(const scalarList& input);

    virtual label size()
    {
        // [UMag, AoA]
        return 2;
    }

    virtual label distributed()
    {
        return 0;
    }
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
