/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v3

    Description:
        Register fluidThermo to Foam::fvMesh

        Example:

        To register a fluidThermo object to fvMesh:

        autoPtr<fluidThermo> fluidThermo = fluidThermo::New(mesh);
        fluidThermo& thermo = fluidThermo();
        DARegDbFluidThermo daThermo(mesh,thermo);

        Then, to get the registered fluidThermo object from fvMesh:

        const DARegDbFluidThermo& daThermo1 = 
            mesh.thisDb().lookupObject<DARegDbFluidThermo>("DARegDbFluidThermo");
        const fluidThermo& thermo1 = daThermo1.getObject();

\*---------------------------------------------------------------------------*/

#ifndef DARegDbFluidThermo_H
#define DARegDbFluidThermo_H

#include "fvOptions.H"
#include "fluidThermo.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                       Class DARegDbFluidThermo Declaration
\*---------------------------------------------------------------------------*/

class DARegDbFluidThermo
    : public regIOobject
{

private:
    /// Disallow default bitwise copy construct
    DARegDbFluidThermo(const DARegDbFluidThermo&);

    /// Disallow default bitwise assignment
    void operator=(const DARegDbFluidThermo&);

    /// Foam::fvMesh object
    const fvMesh& mesh_;

    /// fluid thermo
    const fluidThermo& fluidThermo_;

public:
    /// Constructors
    DARegDbFluidThermo(
        const fvMesh& mesh,
        const fluidThermo& fluidThermo);

    /// Destructor
    virtual ~DARegDbFluidThermo();

    /// return fluidThermo
    const fluidThermo& getObject() const
    {
        return fluidThermo_;
    }

    /// this is a virtual function for regIOobject
    bool writeData(Ostream& os) const;
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
