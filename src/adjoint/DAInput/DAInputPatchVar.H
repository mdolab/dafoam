/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

    Description:
        Child class for a selected variable at a select patch

\*---------------------------------------------------------------------------*/

#ifndef DAInputPatchVar_H
#define DAInputPatchVar_H

#include "DAInput.H"
#include "addToRunTimeSelectionTable.H"
#include "mixedFvPatchFields.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
      Class DAInputPatchVar Declaration
\*---------------------------------------------------------------------------*/

class DAInputPatchVar
    : public DAInput
{

protected:
    /// the name of the variable
    word varName_;

    /// the type of the variable can be either scalar or vector
    word varType_;

public:
    TypeName("patchVar");
    // Constructors

    //- Construct from components
    DAInputPatchVar(
        const word inputName,
        const word inputType,
        fvMesh& mesh,
        const DAOption& daOption,
        const DAModel& daModel,
        const DAIndex& daIndex);

    //- Destructor
    virtual ~DAInputPatchVar()
    {
    }

    virtual void run(const scalarList& input);

    virtual label size()
    {
        if (varType_ == "scalar")
        {
            return 1;
        }
        else if (varType_ == "vector")
        {
            return 3;
        }
        else
        {
            FatalErrorIn("DAInputPatchVar::size") << "varType not valid" << exit(FatalError);
        }
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
