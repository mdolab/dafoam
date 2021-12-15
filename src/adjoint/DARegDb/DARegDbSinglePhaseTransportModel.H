/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v2

    Description:
        Register singlePhaseTransportModel to Foam::fvMesh

        Example:

        To register a singlePhaseTransportModel object to fvMesh:

        singlePhaseTransportModel laminarTransport(U, phi);
        DARegDbSinglePhaseTransportModel sg(mesh,laminarTransport);

        Then, to get the registered nearWallDist object from fvMesh:

        const DARegDbSinglePhaseTransportModel& sg1= 
            mesh.thisDb().lookupObject<DARegDbSinglePhaseTransportModel>(
                "DARegDbSinglePhaseTransportModel");
        const singlePhaseTransportModel& lt1 = sg1.getObject();

\*---------------------------------------------------------------------------*/

#ifndef DARegDbSinglePhaseTransportModel_H
#define DARegDbSinglePhaseTransportModel_H

#include "fvOptions.H"
#include "singlePhaseTransportModel.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                       Class DARegDbSinglePhaseTransportModel Declaration
\*---------------------------------------------------------------------------*/

class DARegDbSinglePhaseTransportModel
    : public regIOobject
{

private:
    /// Disallow default bitwise copy construct
    DARegDbSinglePhaseTransportModel(const DARegDbSinglePhaseTransportModel&);

    /// Disallow default bitwise assignment
    void operator=(const DARegDbSinglePhaseTransportModel&);

    /// Foam::fvMesh object
    const fvMesh& mesh_;

    /// fluid thermo
    const singlePhaseTransportModel& singlePhaseTransportModel_;

public:
    /// Constructors
    DARegDbSinglePhaseTransportModel(
        const fvMesh& mesh,
        const singlePhaseTransportModel& singlePhaseTransportModel);

    /// Destructor
    virtual ~DARegDbSinglePhaseTransportModel();

    /// return fluidThermo
    const singlePhaseTransportModel& getObject() const
    {
        return singlePhaseTransportModel_;
    }

    /// this is a virtual function for regIOobject
    bool writeData(Ostream& os) const;
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
