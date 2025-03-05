/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

    Description:
        Compute the residual for a given solver

\*---------------------------------------------------------------------------*/

#ifndef DAResidual_H
#define DAResidual_H

#include "runTimeSelectionTables.H"
#include "fvOptions.H"
#include "surfaceFields.H"
#include "DAOption.H"
#include "DAModel.H"
#include "DAMacroFunctions.H"
#include "DAUtility.H"
#include "DAIndex.H"
#include "DAField.H"
#include "DAFvSource.H"
#include "IOMRFZoneListDF.H"
#include "constrainHbyA.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                    Class DAResidual Declaration
\*---------------------------------------------------------------------------*/

class DAResidual
    : public regIOobject
{

private:
    /// Disallow default bitwise copy construct
    DAResidual(const DAResidual&);

    /// Disallow default bitwise assignment
    void operator=(const DAResidual&);

protected:
    /// fvMesh
    const fvMesh& mesh_;

    /// DAOption object
    const DAOption& daOption_;

    /// DAModel object
    const DAModel& daModel_;

    /// DAIndex
    const DAIndex& daIndex_;

    /// DAField object
    DAField daField_;

public:
    /// Runtime type information
    TypeName("DAResidual");

    // Declare run-time constructor selection table
    declareRunTimeSelectionTable(
        autoPtr,
        DAResidual,
        dictionary,
        (const word modelType,
         const fvMesh& mesh,
         const DAOption& daOption,
         const DAModel& daModel,
         const DAIndex& daIndex),
        (modelType, mesh, daOption, daModel, daIndex));

    // Constructors

    //- Construct from components
    DAResidual(
        const word modelType,
        const fvMesh& mesh,
        const DAOption& daOption,
        const DAModel& daModel,
        const DAIndex& daIndex);

    // Selectors

    //- Return a reference to the selected model
    static autoPtr<DAResidual> New(
        const word modelType,
        const fvMesh& mesh,
        const DAOption& daOption,
        const DAModel& daModel,
        const DAIndex& daIndex);

    //- Destructor
    virtual ~DAResidual()
    {
    }

    // Members

    /// clear the members
    virtual void clear() = 0;

    /// compute residuals
    virtual void calcResiduals(const dictionary& options) = 0;

    /// update any intermdiate variables that are dependent on state variables and are used in calcResiduals
    virtual void updateIntermediateVariables() = 0;

    /// update the boundary condition for all the states in the selected solver
    virtual void correctBoundaryConditions() = 0;

    /// the master function that compute the residual vector given the state and point vectors
    void masterFunction(
        const dictionary& options,
        const Vec xvVec,
        const Vec wVec,
        Vec resVec);

    /// calculating the adjoint preconditioner matrix using fvMatrix
    virtual void calcPCMatWithFvMatrix(Mat PCMat);

    /// virtual function for regIOobject
    bool writeData(Ostream& os) const
    {
        return true;
    }
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
