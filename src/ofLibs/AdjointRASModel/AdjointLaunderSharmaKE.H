/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v1.0

    Description:
    Child class for the LaunderSharmaKE model
    
\*---------------------------------------------------------------------------*/

#ifndef AdjointLaunderSharmaKE_H
#define AdjointLaunderSharmaKE_H

#include "AdjointRASModel.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                       Class AdjointLaunderSharmaKE Declaration
\*---------------------------------------------------------------------------*/

class AdjointLaunderSharmaKE
:
public AdjointRASModel
{

protected:

    /// \name KE parameters    
    //@{
    dimensionedScalar Cmu_;
    dimensionedScalar C1_;
    dimensionedScalar C2_;
    dimensionedScalar C3_;
    dimensionedScalar sigmak_;
    dimensionedScalar sigmaEps_;
    //@}
        
    /// \name KE functions   
    //@{
    tmp<volScalarField> fMu() const;
    tmp<volScalarField> f2() const;
    tmp<fvScalarMatrix> kSource() const;
    tmp<fvScalarMatrix> epsilonSource() const;
    //- Return the effective diffusivity for k
    tmp<volScalarField> DkEff() const;
    //- Return the effective diffusivity for epsilon
    tmp<volScalarField> DepsilonEff() const;
    //@}

    /// \name Augmented variables for adjoint residuals
    //@{ 
    volScalarField& epsilon_;
    volScalarField epsilonRes_;
    volScalarField epsilonResRef_;
    volScalarField epsilonResPartDeriv_;
    volScalarField epsilonRef_;
    volScalarField& k_;
    volScalarField kRes_;
    volScalarField kResRef_;
    volScalarField kResPartDeriv_;
    volScalarField kRef_;
    //@}
 
public:

    TypeName ("LaunderSharmaKE");
    // Constructors

    //- Construct from components
    AdjointLaunderSharmaKE
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
    );

    //- Destructor
    virtual ~AdjointLaunderSharmaKE()
    {}
    
    
    // Member functions
    
    virtual void updateNut();

    virtual void copyTurbStates(const word option);

    virtual void correctTurbBoundaryConditions();
    
    virtual void calcTurbResiduals
    (  
        const label isRef,
        const label isPC,
        const word fvMatrixName="None"
    );
    
    virtual void correctAdjStateResidualTurbCon
    (
        List< List<word> >& adjStateResidualConInfo
    );
    
    virtual void setAdjStateResidualTurbCon
    (
        HashTable< List< List<word> > >& adjStateResidualConInfo
    );

    virtual void clearTurbVars();

    virtual void writeTurbStates();

};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
