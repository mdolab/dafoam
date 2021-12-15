/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v2

    Description:
        kOmegaSST turbulence model with a multiplicative betaFieldInversion
        field.

\*---------------------------------------------------------------------------*/

#ifndef kOmegaFieldInversion_H
#define kOmegaFieldInversion_H

#include "RASModel.H"
#include "eddyViscosity.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace RASModels
{

/*---------------------------------------------------------------------------*\
                      Class kOmegaFieldInversion Declaration
\*---------------------------------------------------------------------------*/

template<class BasicTurbulenceModel>
class kOmegaFieldInversion
    : public eddyViscosity<RASModel<BasicTurbulenceModel>>
{
    // Private Member Functions

    // Disallow default bitwise copy construct and assignment
    kOmegaFieldInversion(const kOmegaFieldInversion&);
    void operator=(const kOmegaFieldInversion&);

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
    TypeName("kOmegaFieldInversion");

    // Constructors

    //- Construct from components
    kOmegaFieldInversion(
        const alphaField& alpha,
        const rhoField& rho,
        const volVectorField& U,
        const surfaceScalarField& alphaRhoPhi,
        const surfaceScalarField& phi,
        const transportModel& transport,
        const word& propertiesName = turbulenceModel::propertiesName,
        const word& type = typeName);

    //- Destructor
    virtual ~kOmegaFieldInversion()
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
#include "kOmegaFieldInversion.C"
#endif

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
