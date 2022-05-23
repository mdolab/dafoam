/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v3

    Description:
        kOmegaSST turbulence model with a multiplicative betaFieldInversion
        field.

\*---------------------------------------------------------------------------*/

#ifndef kEpsilonFieldInversionEpsilon_H
#define kEpsilonFieldInversionEpsilon_H

#include "RASModel.H"
#include "eddyViscosity.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace RASModels
{

/*---------------------------------------------------------------------------*\
                      Class kEpsilonFieldInversionEpsilon Declaration
\*---------------------------------------------------------------------------*/

template<class BasicTurbulenceModel>
class kEpsilonFieldInversionEpsilon
    : public eddyViscosity<RASModel<BasicTurbulenceModel>>
{
    // Private Member Functions

    // Disallow default bitwise copy construct and assignment
    kEpsilonFieldInversionEpsilon(const kEpsilonFieldInversionEpsilon&);
    void operator=(const kEpsilonFieldInversionEpsilon&);

protected:
    // Protected data

    // Fields

    /// turbulence variables for k-omega SST
    volScalarField omega_;

    volScalarField k_; 

    /// beta field that will be multiplied to the omega production term
    volScalarField betaFieldInversion_;

    /// the prior beta field, usually 1
    volScalarField betaRefFieldInversion_;

    /// the reference velocity field, usually from experiment or higher fidelity sims
    volVectorField varRefFieldInversion_;

    volScalarField profileRefFieldInversion_; 

    volScalarField surfaceFriction_;

    volScalarField surfaceFrictionRef_; 

    volScalarField surfacePressureRef_; 


    //- Wall distance
    //  Note: different to wall distance in parent RASModel
    //  which is for near-wall cells only
    const volScalarField& y_;


    virtual void correctNut();

public:
    typedef typename BasicTurbulenceModel::alphaField alphaField;
    typedef typename BasicTurbulenceModel::rhoField rhoField;
    typedef typename BasicTurbulenceModel::transportModel transportModel;

    //- Runtime type information
    TypeName("kEpsilonFieldInversionEpsilon");

    // Constructors

    //- Construct from components
    kEpsilonFieldInversionEpsilon(
        const alphaField& alpha,
        const rhoField& rho,
        const volVectorField& U,
        const surfaceScalarField& alphaRhoPhi,
        const surfaceScalarField& phi,
        const transportModel& transport,
        const word& propertiesName = turbulenceModel::propertiesName,
        const word& type = typeName);

    //- Destructor
    virtual ~kEpsilonFieldInversionEpsilon()
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
#include "kEpsilonFieldInversionEpsilon.C"
#endif

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
