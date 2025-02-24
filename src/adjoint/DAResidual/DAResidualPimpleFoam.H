/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

    Description:
        Child class for DAPimpleFoam

\*---------------------------------------------------------------------------*/

#ifndef DAResidualPimpleFoam_H
#define DAResidualPimpleFoam_H

#include "DAResidual.H"
#include "addToRunTimeSelectionTable.H"
#include "pimpleControl.H"
#include "adjustPhi.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
      Class DAResidualPimpleFoam Declaration
\*---------------------------------------------------------------------------*/

class DAResidualPimpleFoam
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

    autoPtr<volScalarField> TResPtr_;
    //@}

    /// fvSource term
    volVectorField& fvSource_;

    /// DATurbulenceModel object
    DATurbulenceModel& daTurb_;

    /// pimpleControl object which will be initialized in this class
    pimpleControl pimple_;

    /// whether to has fvSource term
    label hasFvSource_ = 0;

    /// whether to include the temperature field
    label hasTField_ = 0;

    scalar Pr_;

    scalar Prt_;

public:
    TypeName("DAPimpleFoam");
    // Constructors

    //- Construct from components
    DAResidualPimpleFoam(
        const word modelType,
        const fvMesh& mesh,
        const DAOption& daOption,
        const DAModel& daModel,
        const DAIndex& daIndex);

    //- Destructor
    virtual ~DAResidualPimpleFoam()
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
