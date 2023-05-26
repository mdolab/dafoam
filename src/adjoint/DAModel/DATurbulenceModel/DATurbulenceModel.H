/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v3

    Description:
    Augmented turbulence model for the adjoint method, including residual 
    calculation functions, etc

    NOTE 1:
    Instead of inheriting from the OpenFOAM turbulence implementation, in 
    RASModel's children, we re-write all the correspndong functions for each 
    turbulence model. This is to avoid using template classes and template 
    functions for all the other classes in DAFoam. The downside is that we 
    need to update all the RASModel's children when upgrading to a new version 
    of OpenFOAM. Hopefully, the turbulence model part does not change too much 
    from version to version so the modification will be minimal. 

    NOTE 2:
    This function will lookup turbulence model object in fvMesh, so make sure
    you initialize a turbulence model BEFORE initialzing DATurbulenceModel

\*---------------------------------------------------------------------------*/

#ifndef DATurbulenceModel_H
#define DATurbulenceModel_H

#include "runTimeSelectionTables.H"
#include "wallDist.H"
#include "nearWallDist.H"
#include "DAUtility.H"
#include "DAOption.H"
#include "DARegDbNearWallDist.H"
#include "DAMacroFunctions.H"
#include "nutkWallFunctionFvPatchScalarField.H" // for yPlus
#include "wallFvPatch.H" // for yPlus

#ifdef IncompressibleFlow
#include "singlePhaseTransportModel.H"
#include "turbulentTransportModel.H"
#include "DARegDbTurbulenceModelIncompressible.H"
#include "DARegDbSinglePhaseTransportModel.H"
#endif

#ifdef CompressibleFlow
#include "fluidThermo.H"
#include "turbulentFluidThermoModel.H"
#include "DARegDbTurbulenceModelCompressible.H"
#include "DARegDbFluidThermo.H"
#endif

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                    Class DATurbulenceModel Declaration
\*---------------------------------------------------------------------------*/

class DATurbulenceModel
    : public regIOobject
{

private:
    //- Disallow default bitwise copy construct
    DATurbulenceModel(const DATurbulenceModel&);

    //- Disallow default bitwise assignment
    void operator=(const DATurbulenceModel&);

protected:
    /// fvMesh
    const fvMesh& mesh_;

    /// DAOption object
    const DAOption& daOption_;

    /// all DAFoam options
    const dictionary& allOptions_;

    /// turbulence viscosity
    volScalarField& nut_;

    /// velocity
    volVectorField& U_;

    /// face flux
    surfaceScalarField& phi_;

    /// phase field
    volScalarField phase_;

    /// phase*phi*density field
    surfaceScalarField& phaseRhoPhi_;

#ifdef IncompressibleFlow
    /// DARegDbSinglePhaseTransportModel reference
    const DARegDbSinglePhaseTransportModel& daRegDbTransport_;
    /// laminar viscosity for incompressible flow
    const singlePhaseTransportModel& laminarTransport_;
    /// DARegDbTurbulenceModelIncompressible reference
    const DARegDbTurbulenceModelIncompressible& daRegDbTurbIncomp_;
    /// turbulence model object
    const incompressible::turbulenceModel& turbulence_;
    /// density field, initialize with ones
    volScalarField rho_;
#endif

#ifdef CompressibleFlow
    /// DARegDbFluidThermo reference
    const DARegDbFluidThermo& daRegDbThermo_;
    /// thermo model for compressible flow
    const fluidThermo& thermo_;
    /// DARegDbTurbulenceModelCompressible reference
    const DARegDbTurbulenceModelCompressible& daRegDbTurbComp_;
    /// turbulence model object
    const compressible::turbulenceModel& turbulence_;
    /// density field, lookup from fvMesh
    volScalarField& rho_;
#endif

    /// turbulence model property dict
    IOdictionary turbDict_;

    /// turbulence model parameters dict
    dictionary coeffDict_;

    /// Lower limit of k
    dimensionedScalar kMin_;

    /// Lower limit of epsilon
    dimensionedScalar epsilonMin_;

    /// Lower limit for omega
    dimensionedScalar omegaMin_;

    /// Lower limit for nuTilda
    dimensionedScalar nuTildaMin_;

    /// Prandtl number
    scalar Pr_;

    /// turbulent Prandtl number
    scalar Prt_ = -9999.0;

public:
    //- Runtime type information
    TypeName("DATurbulenceModel");

    // Declare run-time constructor selection table
    declareRunTimeSelectionTable(
        autoPtr,
        DATurbulenceModel,
        dictionary,
        (const word modelType,
         const fvMesh& mesh,
         const DAOption& daOption),
        (modelType, mesh, daOption));

    // Constructors

    //- Construct from components
    DATurbulenceModel(
        const word modelType,
        const fvMesh& mesh,
        const DAOption& daOption);

    // Selectors

    //- Return a reference to the selected model
    static autoPtr<DATurbulenceModel> New(
        const word modelType,
        const fvMesh& mesh,
        const DAOption& daOption);

    //- Destructor
    virtual ~DATurbulenceModel()
    {
    }

    // Members

    /// update wall distance for d_. Note: y_ will be automatically updated in mesh_ object
    void correctWallDist();

    /// update alphat
    void correctAlphat();

    /// update nut based on other turbulence variables and update the BCs
    virtual void correctNut() = 0;

    /// update the turbulence state for DAStateInfo::regStates_
    virtual void correctModelStates(wordList& modelStates) const = 0;

    /// update turbulence variable boundary values
    virtual void correctBoundaryConditions() = 0;

    /// update any intermediate variables that are dependent on state variables and are used in calcResiduals
    virtual void updateIntermediateVariables() = 0;

    /// update the original variable connectivity for the adjoint state residuals in stateCon
    virtual void correctStateResidualModelCon(List<List<word>>& stateCon) const = 0;

    /// add the model residual connectivity to stateCon
    virtual void addModelResidualCon(HashTable<List<List<word>>>& allCon) const = 0;

    /// compute the turbulence residuals
    virtual void calcResiduals(const dictionary& options) = 0;

    /// solve the residual equations and update the state
    virtual void correct() = 0;

    /// return the value of the production term from the Spalart Allmaras model
    virtual void getTurbProdTerm(scalarList& prodTerm) const;

    /// dev terms
    tmp<volSymmTensorField> devRhoReff() const;

    /// divDev terms
    tmp<fvVectorMatrix> divDevRhoReff(volVectorField& U);

    /// divDev terms
    tmp<fvVectorMatrix> divDevReff(volVectorField& U);

    /// return effective viscosity
    tmp<volScalarField> nuEff() const;

    /// get the nut field
    tmp<volScalarField> getNut()
    {
        return nut_;
    }

    /// return effective thermal diffusivity
    tmp<volScalarField> alphaEff();

    /// get the nu field
    tmp<volScalarField> nu() const;

    /// get alpha field
    tmp<volScalarField> getAlpha() const;

    /// get the density field
    tmp<volScalarField> getRho()
    {
        return rho_;
    }

    /// get the phase field
    tmp<volScalarField> getPhase()
    {
        return phase_;
    }

    /// get the turbulent Prandtl number
    scalar getPrt()
    {
        if (Prt_ > 0)
        {
            return Prt_;
        }
        else
        {
            // alphat not found in Db so Prt is not set
            FatalErrorIn("getPrt") << exit(FatalError);
            return -9999.0;
        }
    }

#ifdef CompressibleFlow
    const fluidThermo& getThermo() const;
#endif

    /// get mu
    tmp<Foam::volScalarField> getMu() const;

    /// this is a virtual function for regIOobject
    bool writeData(Ostream& os) const;

    /// print the min max and mean yPlus to screen
    void printYPlus() const;

    label isPrintTime(
        const Time& runTime,
        const label printInterval) const;

    /// Inverse transpose product, M_nuTilda^(-T)
    virtual void invTranProdNuTildaEqn(
        const volScalarField& mySource,
        volScalarField& pseudoNuTilda);

    virtual void constructPseudoNuTildaEqn();

    virtual void rhsSolvePseudoNuTildaEqn(const volScalarField& nuTildaSource);

    /// calculate the turbulence residual using LDU matrix
    virtual void calcLduResidualTurb(volScalarField& nuTildaRes);
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
