/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

    Description:
        Child class for residual output

\*---------------------------------------------------------------------------*/

#ifndef DAOutputResidual_H
#define DAOutputResidual_H

#include "DAOutput.H"
#include "addToRunTimeSelectionTable.H"
#include "mixedFvPatchFields.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
      Class DAOutputResidual Declaration
\*---------------------------------------------------------------------------*/

class DAOutputResidual
    : public DAOutput
{

protected:
public:
    TypeName("residual");
    // Constructors

    //- Construct from components
    DAOutputResidual(
        const word outputName,
        fvMesh& mesh,
        const DAOption& daOption,
        DAModel& daModel,
        const DAIndex& daIndex,
        DAResidual& daResidual,
        UPtrList<DAFunction>& daFunctionList);

    //- Destructor
    virtual ~DAOutputResidual()
    {
    }

    virtual void run(scalarList& output);
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
