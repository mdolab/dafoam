/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v1.0

    Description:
    Child class for the v2f model

\*---------------------------------------------------------------------------*/

#ifndef Adjointv2f_H
#define Adjointv2f_H

#include "AdjointRASModel.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                       Class Adjointv2f Declaration
\*---------------------------------------------------------------------------*/

class Adjointv2f
:
public AdjointRASModel
{

protected:

    /// \name v2f parameters    
    //@{
    dimensionedScalar Cmu_;
    dimensionedScalar CmuKEps_;
    dimensionedScalar C1_;
    dimensionedScalar C2_;
    dimensionedScalar CL_;
    dimensionedScalar Ceta_;
    dimensionedScalar Ceps2_;
    dimensionedScalar Ceps3_;
    dimensionedScalar sigmaK_;
    dimensionedScalar sigmaEps_;
    
    //@}
        
    /// \name v2f functions   
    //@{
    //- Return the effective diffusivity for k
    tmp<volScalarField> DkEff() const;
    //- Return the effective diffusivity for epsilon
    tmp<volScalarField> DepsilonEff() const;
    //- Return time scale, Ts
    tmp<volScalarField> Ts() const;
    //- Return length scale, Ls
    tmp<volScalarField> Ls() const;
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
    volScalarField& v2_;
    volScalarField v2Res_;
    volScalarField v2ResRef_;
    volScalarField v2ResPartDeriv_;
    volScalarField v2Ref_;
    volScalarField& f_;
    volScalarField fRes_;
    volScalarField fResRef_;
    volScalarField fResPartDeriv_;
    volScalarField fRef_;
    //@}

    dimensionedScalar v2Min_;
    dimensionedScalar fMin_;

    /// cell-center epsilon values near the wall, this is to fix the issue that the 
    /// epsilonWallFunction will try to modify epsilon values for the cells near walls
    /// this will cause issue for FD-based partial derivatives, so here we basically 
    /// implement a zeroGradient BC for near wall epsilon.
    scalarList epsilonNearWall_;
 
public:

    TypeName ("v2f");
    // Constructors

    //- Construct from components
    Adjointv2f
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
    virtual ~Adjointv2f()
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

    /// save near wall epsilon values to epsilonNearWall_    
    void saveEpsilonNearWall();

    /// set epsilonNearWall_ to near wall epsilon values
    void setEpsilonNearWall();

    /// specially treatment to correct epsilon BC
    void correctEpsilonBoundaryConditions();

    virtual void clearTurbVars();

    virtual void writeTurbStates();

};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
