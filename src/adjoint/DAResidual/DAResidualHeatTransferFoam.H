/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

    Description:
        Child class for DAHeatTransferFoam

\*---------------------------------------------------------------------------*/

#ifndef DAResidualHeatTransferFoam_H
#define DAResidualHeatTransferFoam_H

#include "DAResidual.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
      Class DAResidualHeatTransferFoam Declaration
\*---------------------------------------------------------------------------*/

class DAResidualHeatTransferFoam
    : public DAResidual
{

protected:
    /// \name These are state variables, state residuals, and partial derivatives
    //@{
    volScalarField& T_;
    volScalarField TRes_;
    volScalarField& fvSource_;
    //@}

    autoPtr<dimensionedScalar> kPtr_;

    label hasFvSource_ = 0;

public:
    TypeName("DAHeatTransferFoam");
    // Constructors

    //- Construct from components
    DAResidualHeatTransferFoam(
        const word modelType,
        const fvMesh& mesh,
        const DAOption& daOption,
        const DAModel& daModel,
        const DAIndex& daIndex);

    //- Destructor
    virtual ~DAResidualHeatTransferFoam()
    {
    }

    // Members

    /// clear the members
    virtual void clear();

    /// compute residual
    virtual void calcResiduals(const dictionary& options);

    /// update any intermediate variables that are dependent on state variables and are used in calcResiduals
    virtual void updateIntermediateVariables();

    /// update the boundary condition for all the states in the selected solver
    virtual void correctBoundaryConditions();
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
