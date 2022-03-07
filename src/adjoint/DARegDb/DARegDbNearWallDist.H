/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v3

    Description:
        Register wall distance to Foam::fvMesh

        Example:

        To register a nearWallDist object to fvMesh:

        nearWallDist& d = const_cast<nearWallDist&>(turbulence->y());
        DARegDbNearWallDist daD(mesh,d);

        Then, to get the registered nearWallDist object from fvMesh:

        const DARegDbNearWallDist& daD1=
            mesh.thisDb().lookupObject<DARegDbNearWallDist>("DARegDbNearWallDist");
        const nearWallDist& d1 = daD1.getObject();

\*---------------------------------------------------------------------------*/

#ifndef DARegDbNearWallDist_H
#define DARegDbNearWallDist_H

#include "fvOptions.H"
#include "nearWallDist.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                       Class DARegDbNearWallDist Declaration
\*---------------------------------------------------------------------------*/

class DARegDbNearWallDist
    : public regIOobject
{

private:
    /// Disallow default bitwise copy construct
    DARegDbNearWallDist(const DARegDbNearWallDist&);

    /// Disallow default bitwise assignment
    void operator=(const DARegDbNearWallDist&);

    /// Foam::fvMesh object
    const fvMesh& mesh_;

    /// nearWallDist
    const nearWallDist& nearWallDist_;

public:
    /// Constructors
    DARegDbNearWallDist(
        const fvMesh& mesh,
        const nearWallDist& nearWallDist);

    /// Destructor
    virtual ~DARegDbNearWallDist();

    /// return nearWallDist_
    const nearWallDist& getObject() const
    {
        return nearWallDist_;
    }

    /// this is a virtual function for regIOobject
    bool writeData(Ostream& os) const;
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
