/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v2

    Description:
        Child class for DASolidDisplacementFoam

\*---------------------------------------------------------------------------*/

#ifndef DAResidualSolidDisplacementFoam_H
#define DAResidualSolidDisplacementFoam_H

#include "DAResidual.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
      Class DAResidualSolidDisplacementFoam Declaration
\*---------------------------------------------------------------------------*/

class DAResidualSolidDisplacementFoam
    : public DAResidual
{

protected:
    /// \name These are state variables, state residuals, and partial derivatives
    //@{
    volVectorField& D_;
    volVectorField DRes_;
    //@}

    volTensorField& gradD_;
    volSymmTensorField& sigmaD_;
    volVectorField& divSigmaExp_;
    volScalarField& lambda_;
    volScalarField& mu_;

    label isTractionDisplacementBC_;

    void updateDAndGradD();

public:
    TypeName("DASolidDisplacementFoam");
    // Constructors

    //- Construct from components
    DAResidualSolidDisplacementFoam(
        const word modelType,
        const fvMesh& mesh,
        const DAOption& daOption,
        const DAModel& daModel,
        const DAIndex& daIndex);

    //- Destructor
    virtual ~DAResidualSolidDisplacementFoam()
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
