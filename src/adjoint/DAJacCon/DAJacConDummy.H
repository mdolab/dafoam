/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v2

    Description:
        Child class for dummy Jacobian connectivity. This is for initializing 
        a DAPartDeriv child class that does not need coloring, e.g., dRdBC
        Basically, we implement nothing here because DAPartDeriv-dRdBC will
        call none of the methods from this dummy child class.

\*---------------------------------------------------------------------------*/

#ifndef DAJacConDummy_H
#define DAJacConDummy_H

#include "DAJacCon.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
      Class DAJacConDummy Declaration
\*---------------------------------------------------------------------------*/

class DAJacConDummy
    : public DAJacCon
{

protected:

public:
    TypeName("dummy");
    // Constructors

    //- Construct from components
    DAJacConDummy(
        const word modelType,
        const fvMesh& mesh,
        const DAOption& daOption,
        const DAModel& daModel,
        const DAIndex& daIndex);

    //- Destructor
    virtual ~DAJacConDummy()
    {
    }

    // Member functions

    /// initialize the state Jacobian connectivity matrix
    virtual void initializeJacCon(const dictionary& options);

    /// assign 1 to all non-zero elements for the Jacobian connecitivyt matrix
    virtual void setupJacCon(const dictionary& options);

    /// clear members in parent and child objects
    virtual void clear();

};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
