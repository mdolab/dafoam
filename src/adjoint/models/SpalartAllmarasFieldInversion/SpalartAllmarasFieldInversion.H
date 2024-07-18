/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v3

    Description:
    SpalartAllmaras turbulence model with Fv3 term and the beta
    NOTE: this is just a dummy turbulence model implementation, just to 
    create the variable nuTilda. We shouldn't call any function in this class
    Instead, we should call the correct function in DAModel/DATurbulenceModel

\*---------------------------------------------------------------------------*/

#ifndef SpalartAllmarasFieldInversion_H
#define SpalartAllmarasFieldInversion_H

#include "RASModel.H"
#include "eddyViscosity.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace RASModels
{

/*---------------------------------------------------------------------------*\
                      Class SpalartAllmarasFieldInversion Declaration
\*---------------------------------------------------------------------------*/

template<class BasicTurbulenceModel>
class SpalartAllmarasFieldInversion
    : public eddyViscosity<RASModel<BasicTurbulenceModel>>
{
    // Private Member Functions

    // Disallow default bitwise copy construct and assignment
    SpalartAllmarasFieldInversion(const SpalartAllmarasFieldInversion&);
    void operator=(const SpalartAllmarasFieldInversion&);

protected:
    // Protected data

    // Fields

    /// turbulence variable for SA
    volScalarField nuTilda_;

    volScalarField betaFieldInversion_;

    volVectorField UData_;

    volScalarField surfaceFriction_;

    volScalarField surfaceFrictionData_; 

    volScalarField pData_; 

    volScalarField USingleComponentData_; 

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
    TypeName("SpalartAllmarasFieldInversion");

    // Constructors

    //- Construct from components
    SpalartAllmarasFieldInversion(
        const alphaField& alpha,
        const rhoField& rho,
        const volVectorField& U,
        const surfaceScalarField& alphaRhoPhi,
        const surfaceScalarField& phi,
        const transportModel& transport,
        const word& propertiesName = turbulenceModel::propertiesName,
        const word& type = typeName);

    //- Destructor
    virtual ~SpalartAllmarasFieldInversion()
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
#include "SpalartAllmarasFieldInversion.C"
#endif

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
