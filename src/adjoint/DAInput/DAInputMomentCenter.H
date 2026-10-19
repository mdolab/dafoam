/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v5

    Description:
        Child class for the moment center [x, y, z] used in DAFunctionMoment

\*---------------------------------------------------------------------------*/

#ifndef DAInputMomentCenter_H
#define DAInputMomentCenter_H

#include "DAInput.H"
#include "addToRunTimeSelectionTable.H"
#include "DAGlobalVar.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
      Class DAInputMomentCenter Declaration
\*---------------------------------------------------------------------------*/

class DAInputMomentCenter
    : public DAInput
{

protected:
public:
    TypeName("momentCenter");
    // Constructors

    //- Construct from components
    DAInputMomentCenter(
        const word inputName,
        const word inputType,
        fvMesh& mesh,
        const DAOption& daOption,
        const DAModel& daModel,
        const DAIndex& daIndex);

    //- Destructor
    virtual ~DAInputMomentCenter()
    {
    }

    virtual void run(const scalarList& input);

    virtual label size()
    {
        // [x, y, z]
        return 3;
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
