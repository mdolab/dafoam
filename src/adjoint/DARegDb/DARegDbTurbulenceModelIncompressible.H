/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v2

    Description:
        Register a turbulence model object to Foam::fvMesh

        Example:

        To register a turbulence model object to fvMesh:

        autoPtr<incompressible::turbulenceModel> turbulencePtr(
            incompressible::turbulenceModel::New(U, phi, laminarTransport));
        incompressible::turbulenceModel& turbulence = turbulencePtr();
        DARegDbTurbulenceModelIncompressible turb(mesh,turbulence);

        Then, to get the registered turbulence model object from fvMesh:

        const DARegDbTurbulenceModelIncompressible& turbDb= 
            mesh.thisDb().lookupObject<DARegDbTurbulenceModelIncompressible>(
                "DARegDbTurbulenceModelIncompressible");
        const incompressible::turbulenceModel& turb1 = turbDb.getObject();

\*---------------------------------------------------------------------------*/

#ifndef DARegDbTurbulenceModelIncompressible_H
#define DARegDbTurbulenceModelIncompressible_H

#include "fvOptions.H"
#include "singlePhaseTransportModel.H"
#include "turbulentTransportModel.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                       Class DARegDbTurbulenceModelIncompressible Declaration
\*---------------------------------------------------------------------------*/

class DARegDbTurbulenceModelIncompressible
    : public regIOobject
{

private:
    /// Disallow default bitwise copy construct
    DARegDbTurbulenceModelIncompressible(const DARegDbTurbulenceModelIncompressible&);

    /// Disallow default bitwise assignment
    void operator=(const DARegDbTurbulenceModelIncompressible&);

    /// Foam::fvMesh object
    const fvMesh& mesh_;

    /// fluid thermo
    const incompressible::turbulenceModel& turbulenceModel_;

public:
    /// Constructors
    DARegDbTurbulenceModelIncompressible(
        const fvMesh& mesh,
        const incompressible::turbulenceModel& turbulenceModel);

    /// Destructor
    virtual ~DARegDbTurbulenceModelIncompressible();

    /// return fluidThermo
    const incompressible::turbulenceModel& getObject() const
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
