/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v2

    Description:
        Child class for DAPisoFoam

\*---------------------------------------------------------------------------*/

#ifndef DAResidualPisoFoam_H
#define DAResidualPisoFoam_H

#include "DAResidual.H"
#include "addToRunTimeSelectionTable.H"
#include "pisoControl.H"
#include "adjustPhi.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
      Class DAResidualPisoFoam Declaration
\*---------------------------------------------------------------------------*/

class DAResidualPisoFoam
    : public DAResidual
{
protected:
    /// \name These are state variables, state residuals, and partial derivatives
    //@{
    volVectorField& U_;
    volVectorField URes_;

    volScalarField& p_;
    volScalarField pRes_;

    surfaceScalarField& phi_;
    surfaceScalarField phiRes_;
    //@}

    /// fvSource term
    volVectorField& fvSource_;

    /// DATurbulenceModel object
    DATurbulenceModel& daTurb_;

    /// pisoControl object which will be initialized in this class
    pisoControl piso_;

    /// whether to has fvSource term
    label hasFvSource_ = 0;

    /// whether the hybrid adjoint or time accurate adjoint is active
    word mode_ = "None";

public:
    TypeName("DAPisoFoam");
    // Constructors

    //- Construct from components
    DAResidualPisoFoam(
        const word modelType,
        const fvMesh& mesh,
        const DAOption& daOption,
        const DAModel& daModel,
        const DAIndex& daIndex);

    //- Destructor
    virtual ~DAResidualPisoFoam()
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
