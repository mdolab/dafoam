/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v3

    Description:
        Compute the rigid body motion

\*---------------------------------------------------------------------------*/

#ifndef DAMotion_H
#define DAMotion_H

#include "runTimeSelectionTables.H"
#include "fvOptions.H"
#include "DAOption.H"
#include "dynamicFvMesh.H"
#include "fvc.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                    Class DAMotion Declaration
\*---------------------------------------------------------------------------*/

class DAMotion
{

private:
    /// Disallow default bitwise copy construct
    DAMotion(const DAMotion&);

    /// Disallow default bitwise assignment
    void operator=(const DAMotion&);

protected:
    /// fvMesh
    const dynamicFvMesh& mesh_;

    /// DAOption object
    const DAOption& daOption_;

    /// patch names for the moving body
    wordList patchNames_;

public:
    /// Runtime type information
    TypeName("DAMotion");

    // Declare run-time constructor selection table
    declareRunTimeSelectionTable(
        autoPtr,
        DAMotion,
        dictionary,
        (
            const dynamicFvMesh& mesh,
            const DAOption& daOption),
        (
            mesh,
            daOption));

    // Constructors

    //- Construct from components
    DAMotion(
        const dynamicFvMesh& mesh,
        const DAOption& daOption);

    // Selectors

    //- Return a reference to the selected model
    static autoPtr<DAMotion> New(
        const dynamicFvMesh& mesh,
        const DAOption& daOption);

    //- Destructor
    virtual ~DAMotion()
    {
    }

    /// update the cell displacement
    virtual void correct() = 0;

    /// calculate the force
    vector getForce(const dynamicFvMesh& mesh);
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
