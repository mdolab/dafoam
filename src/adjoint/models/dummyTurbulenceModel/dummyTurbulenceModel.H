/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

    Description:
        A dummy turbulence model that mimics the option for RAS-laminar flow

\*---------------------------------------------------------------------------*/

#ifndef dummyTurbulenceModel_H
#define dummyTurbulenceModel_H

#include "RASModel.H"
#include "eddyViscosity.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace RASModels
{

/*---------------------------------------------------------------------------*\
                      Class dummyTurbulenceModel Declaration
\*---------------------------------------------------------------------------*/

template<class BasicTurbulenceModel>
class dummyTurbulenceModel
    : public eddyViscosity<RASModel<BasicTurbulenceModel>>
{
    // Private Member Functions

    // Disallow default bitwise copy construct and assignment
    dummyTurbulenceModel(const dummyTurbulenceModel&);
    void operator=(const dummyTurbulenceModel&);

protected:
    virtual void correctNut();

public:
    typedef typename BasicTurbulenceModel::alphaField alphaField;
    typedef typename BasicTurbulenceModel::rhoField rhoField;
    typedef typename BasicTurbulenceModel::transportModel transportModel;

    //- Runtime type information
    TypeName("dummy");

    // Constructors

    //- Construct from components
    dummyTurbulenceModel(
        const alphaField& alpha,
        const rhoField& rho,
        const volVectorField& U,
        const surfaceScalarField& alphaRhoPhi,
        const surfaceScalarField& phi,
        const transportModel& transport,
        const word& propertiesName = turbulenceModel::propertiesName,
        const word& type = typeName);

    //- Destructor
    virtual ~dummyTurbulenceModel()
    {
    }

    // Member Functions

    //- Re-read model coefficients if they have changed
    virtual bool read();

    //- Return the turbulence kinetic energy
    virtual tmp<volScalarField> k() const;

    //- Return the turbulence kinetic energy dissipation rate
    virtual tmp<volScalarField> epsilon() const;

    //- Solve the turbulence equations and correct the turbulence viscosity
    virtual void correct();
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace RASModels
} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#ifdef NoRepository
#include "dummyTurbulenceModel.C"
#endif

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
