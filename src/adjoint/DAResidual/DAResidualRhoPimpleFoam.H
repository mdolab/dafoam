/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

    Description:
        Child class for DARhoPimpleFoam

\*---------------------------------------------------------------------------*/

#ifndef DAResidualRhoPimpleFoam_H
#define DAResidualRhoPimpleFoam_H

#include "DAResidual.H"
#include "addToRunTimeSelectionTable.H"
#include "pimpleControl.H"
#include "adjustPhi.H"
#include "thermodynamicConstants.H"
#include "pressureControl.H"
#include "constrainPressure.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
      Class DAResidualRhoPimpleFoam Declaration
\*---------------------------------------------------------------------------*/

class DAResidualRhoPimpleFoam
    : public DAResidual
{

protected:
    /// \name These are state variables, state residuals, and partial derivatives
    //@{
    volVectorField& U_;
    volVectorField URes_;

    volScalarField& p_;
    volScalarField pRes_;

    volScalarField& T_;
    volScalarField TRes_;

    surfaceScalarField& phi_;
    surfaceScalarField phiRes_;
    //@}

    /// fvSource term
    volVectorField& fvSource_;

    /// fvSource term for the energy equation
    volScalarField& fvSourceEnergy_;

    /// thermophysical property
    fluidThermo& thermo_;

    /// \name These are intermediate variables
    //@{
    volScalarField& he_;
    volScalarField& rho_;
    volScalarField& alphat_;
    volScalarField& psi_;
    volScalarField& dpdt_;
    volScalarField& K_;
    //@}

    /// DATurbulenceModel object
    DATurbulenceModel& daTurb_;

    /// pimpleControl object which will be initialized in this class
    pimpleControl pimple_;

    /// \name These are constants to update the intermediate variables
    //@{
    scalar molWeight_;
    scalar Cp_;
    //@}

    /// whether to have fvSource term
    label hasFvSource_ = 0;

public:
    TypeName("DARhoPimpleFoam");
    // Constructors

    //- Construct from components
    DAResidualRhoPimpleFoam(
        const word modelType,
        const fvMesh& mesh,
        const DAOption& daOption,
        const DAModel& daModel,
        const DAIndex& daIndex);

    //- Destructor
    virtual ~DAResidualRhoPimpleFoam()
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

    virtual void calcPCMatWithFvMatrix(Mat PCMat);
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
