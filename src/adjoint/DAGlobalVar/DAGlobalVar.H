/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

    Description:
        This class saves global variables and register it to mesh.thisDb()
        such that all other class can access its members

\*---------------------------------------------------------------------------*/

#ifndef DAGlobalVar_H
#define DAGlobalVar_H

#include "fvOptions.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                       Class DAGlobalVar Declaration
\*---------------------------------------------------------------------------*/

class DAGlobalVar
    : public regIOobject
{

private:
    /// Disallow default bitwise copy construct
    DAGlobalVar(const DAGlobalVar&);

    /// Disallow default bitwise assignment
    void operator=(const DAGlobalVar&);

protected:
    /// Foam::fvMesh object
    const fvMesh& mesh_;

public:
    /// Constructors
    DAGlobalVar(const fvMesh& mesh);

    /// Destructor
    virtual ~DAGlobalVar()
    {
    }

    /// this is a virtual function for regIOobject
    bool writeData(Ostream& os) const
    {
        return true;
    }

    // ************** global variables ******************

    /// patch velocity list [UMag, AOA] AoA is in degrees
    scalarList patchVelocity = {0.0, 0.0};

    /// the runStatus can be either primal or adjoint. Other classes can use this info
    word runStatus = "primal";
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
