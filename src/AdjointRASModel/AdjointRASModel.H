/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v1.0

    Description:
    Augmented RASModel for the adjoint method, including residual calculation functions, etc

    NOTE:
    Instead of inheriting from the OpenFOAM turbulence implementation, 
    in RASModel's children, we re-write all the correspndong functions for each turbulence model . 
    This is to avoid using template classes and template functions for all the other classes in DAFoam. 
    The downside is that we need to update all the RASModel's children when upgrading to a new version 
    of OpenFOAM. Hopefully, the turbulence model part does not change too much from version to version 
    so the modification will be minimal. 

\*---------------------------------------------------------------------------*/

#ifndef AdjointRASModel_H
#define AdjointRASModel_H

#include "runTimeSelectionTables.H"
#include "fvOptions.H"
#include "surfaceFields.H"
#include "AdjointIO.H"
#include "fvCFD.H"
#include "geometricOneField.H"
#include "wallDist.H"
#include "turbulenceModel.H"
#include "nearWallDist.H"
#include "bound.H"

#ifdef IncompressibleFlow
    #include "singlePhaseTransportModel.H"
#endif

#ifdef CompressibleFlow
    #include "fluidThermo.H"
#endif

#ifdef PhaseIncompressibleFlow
    #include "immiscibleIncompressibleTwoPhaseMixture.H"
#endif

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                    Class AdjointRASModel Declaration
\*---------------------------------------------------------------------------*/

class AdjointRASModel
{

private:

    //- Disallow default bitwise copy construct
    AdjointRASModel(const AdjointRASModel&);

    //- Disallow default bitwise assignment
    void operator=(const AdjointRASModel&);

protected:

    const fvMesh& mesh_;
    const AdjointIO& adjIO_;
    /// near wall distance for patches
    nearWallDist& d_;
    const objectRegistry& db_;
    
    volScalarField& nut_;
    volVectorField& U_;
    surfaceScalarField& phi_;

#ifdef IncompressibleFlow
    /// laminar transport model for incompressible flow
    const singlePhaseTransportModel& laminarTransport_;
    /// unity field for multiphase flow cases
    volScalarField phase_;
    /// unity field for rho
    volScalarField rho_;
    /// phi field
    surfaceScalarField& phaseRhoPhi_;
#endif

#ifdef CompressibleFlow
    /// thermophysical model for compressible flow
    const fluidThermo& thermo_;
    /// unity field for multiphase flow cases
    volScalarField phase_;
    /// using existing rho field
    volScalarField& rho_;
    /// phi field
    surfaceScalarField& phaseRhoPhi_;
#endif

#ifdef PhaseIncompressibleFlow
    /// thermophysical model for compressible flow
    const immiscibleIncompressibleTwoPhaseMixture& mixture_;
    /// unity field for multiphase flow cases
    volScalarField phase_;
    /// using existing rho field
    volScalarField& rho_;
    /// phi field
    surfaceScalarField& phaseRhoPhi_;
#endif

    IOdictionary turbDict_;
    dictionary coeffDict_;

    /// Lower limit of k
    dimensionedScalar kMin_;

    /// Lower limit of epsilon
    dimensionedScalar epsilonMin_;

    /// Lower limit for omega
    dimensionedScalar omegaMin_;

    /// Lower limit for nuTilda
    dimensionedScalar nuTildaMin_;

public:

    //- Runtime type information
    TypeName("AdjointRASModel");

    // Declare run-time constructor selection table
    declareRunTimeSelectionTable
    (
        autoPtr,
        AdjointRASModel,
        dictionary,
        (
            const fvMesh& mesh,
            const AdjointIO& adjIO,
            nearWallDist& d,
#ifdef IncompressibleFlow
            const singlePhaseTransportModel& laminarTransport
#endif
#ifdef CompressibleFlow
            const fluidThermo& thermo
#endif
#ifdef PhaseIncompressibleFlow
            const immiscibleIncompressibleTwoPhaseMixture& mixture
#endif
        ),
#ifdef IncompressibleFlow
        (mesh,adjIO,d,laminarTransport)
#endif
#ifdef CompressibleFlow
        (mesh,adjIO,d,thermo)
#endif
#ifdef PhaseIncompressibleFlow
        (mesh,adjIO,d,mixture)
#endif
    );


    // Constructors

    //- Construct from components
    AdjointRASModel
    (
        const fvMesh& mesh,
        const AdjointIO& adjIO,
        nearWallDist& d,
#ifdef IncompressibleFlow
        const singlePhaseTransportModel& laminarTransport
#endif
#ifdef CompressibleFlow
        const fluidThermo& thermo
#endif
#ifdef PhaseIncompressibleFlow
        const immiscibleIncompressibleTwoPhaseMixture& mixture
#endif
    );


    // Selectors

    //- Return a reference to the selected model
    static autoPtr<AdjointRASModel> New
    (
        const fvMesh& mesh,
        const AdjointIO& adjIO,
        nearWallDist& d,
#ifdef IncompressibleFlow
        const singlePhaseTransportModel& laminarTransport
#endif
#ifdef CompressibleFlow
        const fluidThermo& thermo
#endif
#ifdef PhaseIncompressibleFlow
        const immiscibleIncompressibleTwoPhaseMixture& mixture
#endif
    );


    //- Destructor
    virtual ~AdjointRASModel()
    {}
    
    // Members

    /// diagonal components of fvMatrix, which fvMatrix to use depends on fvMatrixName in calcTurbResiduals function 
    scalarList fvMatrixDiag;

    /// lower components of fvMatrix, which fvMatrix to use depends on fvMatrixName in calcTurbResiduals function
    scalarList fvMatrixLower;

    /// upper component of fvMatrix, which fvMatrix to use depends on fvMatrixName in calcTurbResiduals function
    scalarList fvMatrixUpper; 
    
    // Member functions

    /// a list of turbulence state variable names
    DynamicList<Foam::string> turbStates;
    
    /// update wall distance for d_. Note: y_ will be automatically updated in mesh_ object
    void correctWallDist();
    
    /// update nut based on other turbulence variables and update the BCs
    virtual void updateNut()=0;
    
    /// copy turbulence state variables to their references or vice versa, depending on option
    virtual void copyTurbStates(const word option)=0;

    /// update turbulence variable boundary values
    virtual void correctTurbBoundaryConditions()=0;
    
    /// calculate turbulence residuals
    virtual void calcTurbResiduals
    (  
        const label isRef,
        const label isPC,
        const word fvMatrixName="None"
    )=0;

    /// replace nut with the correct turbulence state variables, depending on the turbulence models
    virtual void correctAdjStateResidualTurbCon
    (
        List< List<word> >& adjStateResidualConInfo
    )=0;
    
    /// setup the turbulence variable connectivity levels for the provided listlist.
    virtual void setAdjStateResidualTurbCon
    (
        HashTable< List< List<word> > >& adjStateResidualConInfo
    )=0;
    
    /// dev terms
    tmp<volSymmTensorField> devRhoReff();
    
    /// divDev terms
    tmp<fvVectorMatrix> divDevRhoReff(volVectorField& U);
    
    /// divDev terms
    tmp<fvVectorMatrix> divDevReff(volVectorField& U);
    
    /// return effective viscosity
    tmp<volScalarField> nuEff();

    /// get the nut field
    tmp<volScalarField> getNut()
    {
        return nut_;
    }

    /// return effective thermal diffusivity
    tmp<volScalarField> alphaEff();

    /// get the nu field
    tmp<volScalarField> getNu() const;

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

    /// get the near wall distance d_    
    scalar getNearWallDist(label patchI, label faceI)
    {
        return d_[patchI][faceI];
    }
    
    /// get mu
    tmp<Foam::volScalarField> getMu() const;

    /// clear turbulence state variable volume fields, this will save memory 
    virtual void clearTurbVars()=0;

    /// write turbulence states to files
    virtual void writeTurbStates()=0;

    /// bound turbulence variable
    void boundTurbVars(volScalarField&, const dimensionedScalar& lowerBound);

};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
