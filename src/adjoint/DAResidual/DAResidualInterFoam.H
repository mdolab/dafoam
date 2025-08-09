/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

    Description:
        Child class for DAInterFoam

\*---------------------------------------------------------------------------*/

#ifndef DAResidualInterFoam_H
#define DAResidualInterFoam_H

#include "DAResidual.H"
#include "addToRunTimeSelectionTable.H"
#include "pimpleControl.H"
#include "adjustPhi.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
      Class DAResidualInterFoam Declaration
\*---------------------------------------------------------------------------*/

class DAResidualInterFoam
    : public DAResidual
{
protected:
public:
    TypeName("DAInterFoam");
    // Constructors

    //- Construct from components
    DAResidualInterFoam(
        const word modelType,
        const fvMesh& mesh,
        const DAOption& daOption,
        const DAModel& daModel,
        const DAIndex& daIndex);

    //- Destructor
    virtual ~DAResidualInterFoam()
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
