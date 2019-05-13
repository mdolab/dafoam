/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v1812

\*---------------------------------------------------------------------------*/

#include "dummyTurbulenceModel.H"
#include "fvOptions.H"
#include "bound.H"
#include "wallDist.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace RASModels
{

// * * * * * * * * * * * * Protected Member Functions  * * * * * * * * * * * //

template<class BasicTurbulenceModel>
void dummyTurbulenceModel<BasicTurbulenceModel>::correctNut()
{
    
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

template<class BasicTurbulenceModel>
dummyTurbulenceModel<BasicTurbulenceModel>::dummyTurbulenceModel
(
    const alphaField& alpha,
    const rhoField& rho,
    const volVectorField& U,
    const surfaceScalarField& alphaRhoPhi,
    const surfaceScalarField& phi,
    const transportModel& transport,
    const word& propertiesName,
    const word& type
)
:
    eddyViscosity<RASModel<BasicTurbulenceModel>>
    (
        type,
        alpha,
        rho,
        U,
        alphaRhoPhi,
        phi,
        transport,
        propertiesName
    )
{

}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

template<class BasicTurbulenceModel>
bool dummyTurbulenceModel<BasicTurbulenceModel>::read()
{
    
    return true;
}

template<class BasicTurbulenceModel>
tmp<volScalarField> dummyTurbulenceModel<BasicTurbulenceModel>::k() const
{
    return this->nut_;
}


template<class BasicTurbulenceModel>
tmp<volScalarField> dummyTurbulenceModel<BasicTurbulenceModel>::epsilon() const
{
    return this->nut_;
}


template<class BasicTurbulenceModel>
void dummyTurbulenceModel<BasicTurbulenceModel>::correct()
{

}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace RASModels
} // End namespace Foam

// ************************************************************************* //
