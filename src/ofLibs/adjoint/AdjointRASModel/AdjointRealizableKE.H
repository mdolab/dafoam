/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v1.0

    Description:
    Child class for the realizableKE model

\*---------------------------------------------------------------------------*/

#ifndef AdjointRealizableKE_H
#define AdjointRealizableKE_H

#include "AdjointRASModel.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                       Class AdjointRealizableKE Declaration
\*---------------------------------------------------------------------------*/

class AdjointRealizableKE
:
public AdjointRASModel
{

protected:

    /// \name KE parameters    
    //@{
    dimensionedScalar A0_;
    dimensionedScalar C2_;
    dimensionedScalar sigmak_;
    dimensionedScalar sigmaEps_;
    //@}
        
    /// \name KE functions   
    //@{
    tmp<fvScalarMatrix> kSource() const;
    tmp<fvScalarMatrix> epsilonSource() const;
    tmp<volScalarField> rCmu
    (
        const volTensorField& gradU,
        const volScalarField& S2,
        const volScalarField& magS
    );
    //- Return the effective diffusivity for k
    tmp<volScalarField> DkEff() const
    {
        return tmp<volScalarField>
        (
            new volScalarField
            (
                "DkEff",
                (this->nut_/sigmak_ + this->getNu())
            )
        );
    }
    //- Return the effective diffusivity for epsilon
    tmp<volScalarField> DepsilonEff() const
    {
        return tmp<volScalarField>
        (
            new volScalarField
            (
                "DepsilonEff",
                (this->nut_/sigmaEps_ + this->getNu())
            )
        );
    }
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

    /// cell-center epsilon values near the wall, this is to fix the issue that the 
    /// epsilonWallFunction will try to modify epsilon values for the cells near walls
    /// this will cause issue for FD-based partial derivatives, so here we basically 
    /// implement a zeroGradient BC for near wall epsilon.
    scalarList epsilonNearWall_;
 
public:

    TypeName ("realizableKE");
    // Constructors

    //- Construct from components
    AdjointRealizableKE
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
    virtual ~AdjointRealizableKE()
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
