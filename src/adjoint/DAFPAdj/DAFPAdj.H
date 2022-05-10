/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v3

    Description:
        Compute object functions and their derivatives

\*---------------------------------------------------------------------------*/

#ifndef DAFPAdj_H
#define DAFPAdj_H

#include "runTimeSelectionTables.H"
#include "fvOptions.H"
#include "DAOption.H"
#include "DAModel.H"
#include "DAIndex.H"
#include "DAField.H"
#include "DAResidual.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                    Class DAFPAdj Declaration
\*---------------------------------------------------------------------------*/

class DAFPAdj
{

private:
    /// Disallow default bitwise copy construct
    DAFPAdj(const DAFPAdj&);

    /// Disallow default bitwise assignment
    void operator=(const DAFPAdj&);

protected:
    /// fvMesh
    const fvMesh& mesh_;

    /// DAOption object
    const DAOption& daOption_;

    /// DAModel object
    const DAModel& daModel_;

    /// DAIndex object
    const DAIndex& daIndex_;

    /// DAResidual object
    const DAResidual& daResidual_;

    /// DAField object
    DAField daField_;

public:
    /// Runtime type information
    TypeName("DAFPAdj");

    // Declare run-time constructor selection table
    declareRunTimeSelectionTable(
        autoPtr,
        DAFPAdj,
        dictionary,
        (
            const word modelType,
            const fvMesh& mesh,
            const DAOption& daOption,
            const DAModel& daModel,
            const DAIndex& daIndex,
            const DAResidual& daResidual),
        (
            modelType,
            mesh,
            daOption,
            daModel,
            daIndex,
            daResidual));

    // Constructors

    //- Construct from components
    DAFPAdj(
        const word modelType,
        const fvMesh& mesh,
        const DAOption& daOption,
        const DAModel& daModel,
        const DAIndex& daIndex,
        const DAResidual& daResidual);

    // Selectors

    //- Return a reference to the selected model
    static autoPtr<DAFPAdj> New(
        const word modelType,
        const fvMesh& mesh,
        const DAOption& daOption,
        const DAModel& daModel,
        const DAIndex& daIndex,
        const DAResidual& daResidual);

    //- Destructor
    virtual ~DAFPAdj()
    {
    }

    /// solve the adjoint using the fixed-point iteration method
    virtual label run(
        Vec dFdW,
        Vec psi) = 0;
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
