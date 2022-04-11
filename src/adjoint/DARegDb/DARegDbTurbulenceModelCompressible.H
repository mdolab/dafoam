/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v3

    Description:
        Register a turbulence model object to Foam::fvMesh

        Example:

        To register a turbulence model object to fvMesh:

        autoPtr<compressible::turbulenceModel> turbulencePtr(
            compressible::turbulenceModel::New(rho, U, phi, thermo));
        compressible::turbulenceModel& turbulence = turbulencePtr();
        DARegDbTurbulenceModelCompressible turb(mesh,turbulence);

        Then, to get the registered turbulence model object from fvMesh:

        const DARegDbTurbulenceModelCompressible& turbDb= 
            mesh.thisDb().lookupObject<DARegDbTurbulenceModelCompressible>(
                "DARegDbTurbulenceModelCompressible");
        const compressible::turbulenceModel& turb1 = turbDb.getObject();

\*---------------------------------------------------------------------------*/

#ifndef DARegDbTurbulenceModelCompressible_H
#define DARegDbTurbulenceModelCompressible_H

#include "fvOptions.H"
#include "fluidThermo.H"
#include "turbulentFluidThermoModel.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                       Class DARegDbTurbulenceModelCompressible Declaration
\*---------------------------------------------------------------------------*/

class DARegDbTurbulenceModelCompressible
    : public regIOobject
{

private:
    /// Disallow default bitwise copy construct
    DARegDbTurbulenceModelCompressible(const DARegDbTurbulenceModelCompressible&);

    /// Disallow default bitwise assignment
    void operator=(const DARegDbTurbulenceModelCompressible&);

    /// Foam::fvMesh object
    const fvMesh& mesh_;

    /// turbulence model object
    const compressible::turbulenceModel& turbulenceModel_;

public:
    /// Constructors
    DARegDbTurbulenceModelCompressible(
        const fvMesh& mesh,
        const compressible::turbulenceModel& turbulenceModel);

    /// Destructor
    virtual ~DARegDbTurbulenceModelCompressible();

    /// return fluidThermo
    const compressible::turbulenceModel& getObject() const
    {
        return turbulenceModel_;
    }

    /// this is a virtual function for regIOobject
    bool writeData(Ostream& os) const;
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
