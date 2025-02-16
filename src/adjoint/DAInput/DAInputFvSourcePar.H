/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

    Description:
        Child class for regression parameters

\*---------------------------------------------------------------------------*/

#ifndef DAInputFvSourcePar_H
#define DAInputFvSourcePar_H

#include "DAInput.H"
#include "addToRunTimeSelectionTable.H"
#include "DAGlobalVar.H"
#include "DAFvSource.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
      Class DAInputFvSourcePar Declaration
\*---------------------------------------------------------------------------*/

class DAInputFvSourcePar
    : public DAInput
{

protected:
    word fvSourceType_;

    word fvSourceName_;

    labelList indices_;

public:
    TypeName("fvSourcePar");
    // Constructors

    //- Construct from components
    DAInputFvSourcePar(
        const word inputName,
        const word inputType,
        fvMesh& mesh,
        const DAOption& daOption,
        const DAModel& daModel,
        const DAIndex& daIndex);

    //- Destructor
    virtual ~DAInputFvSourcePar()
    {
    }

    virtual void run(const scalarList& input);

    virtual label size()
    {
        return indices_.size();
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
