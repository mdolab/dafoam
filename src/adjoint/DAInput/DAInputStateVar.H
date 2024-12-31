/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

    Description:
        Child class for state variable input

\*---------------------------------------------------------------------------*/

#ifndef DAInputStateVar_H
#define DAInputStateVar_H

#include "DAInput.H"
#include "addToRunTimeSelectionTable.H"
#include "mixedFvPatchFields.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
      Class DAInputStateVar Declaration
\*---------------------------------------------------------------------------*/

class DAInputStateVar
    : public DAInput
{

protected:
public:
    TypeName("stateVar");
    // Constructors

    //- Construct from components
    DAInputStateVar(
        const word inputName,
        const word inputType,
        fvMesh& mesh,
        const DAOption& daOption,
        const DAModel& daModel,
        const DAIndex& daIndex);

    //- Destructor
    virtual ~DAInputStateVar()
    {
    }

    virtual void run(const scalarList& input);

    virtual label size()
    {
        return daIndex_.nLocalAdjointStates;
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
