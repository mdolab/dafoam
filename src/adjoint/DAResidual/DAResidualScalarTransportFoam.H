/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v2

    Description:
        Child class for DAScalarTransportFoam

\*---------------------------------------------------------------------------*/

#ifndef DAResidualScalarTransportFoam_H
#define DAResidualScalarTransportFoam_H

#include "DAResidual.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
      Class DAResidualScalarTransportFoam Declaration
\*---------------------------------------------------------------------------*/

class DAResidualScalarTransportFoam
    : public DAResidual
{

protected:
    /// \name These are state variables, state residuals, and partial derivatives
    //@{
    volScalarField& T_;
    volScalarField TRes_;
    surfaceScalarField& phi_;
    //@}

    scalar DT_;

public:
    TypeName("DAScalarTransportFoam");
    // Constructors

    //- Construct from components
    DAResidualScalarTransportFoam(
        const word modelType,
        const fvMesh& mesh,
        const DAOption& daOption,
        const DAModel& daModel,
        const DAIndex& daIndex);

    //- Destructor
    virtual ~DAResidualScalarTransportFoam()
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
