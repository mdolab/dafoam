/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v1.0

    Description:
    Child class for the kOmegaSST model

\*---------------------------------------------------------------------------*/

#ifndef AdjointkOmegaSST_H
#define AdjointkOmegaSST_H

#include "AdjointRASModel.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                       Class AdjointkOmegaSST Declaration
\*---------------------------------------------------------------------------*/

class AdjointkOmegaSST
:
public AdjointRASModel
{

protected:

    /// \name SST parameters    
    //@{
    dimensionedScalar alphaK1_;
    dimensionedScalar alphaK2_;

    dimensionedScalar alphaOmega1_;
    dimensionedScalar alphaOmega2_;

    dimensionedScalar gamma1_;
    dimensionedScalar gamma2_;

    dimensionedScalar beta1_;
    dimensionedScalar beta2_;

    dimensionedScalar betaStar_;

    dimensionedScalar a1_;
    dimensionedScalar b1_;
    dimensionedScalar c1_;

    Switch F3_;
    //@}
        
    /// \name SST functions   
    //@{
    tmp<volScalarField> F1(const volScalarField& CDkOmega) const;
    tmp<volScalarField> F2() const;
    tmp<volScalarField> F3() const;
    tmp<volScalarField> F23() const;

    tmp<volScalarField> blend
    (
        const volScalarField& F1,
        const dimensionedScalar& psi1,
        const dimensionedScalar& psi2
    ) const
    {
        return F1*(psi1 - psi2) + psi2;
    }

    tmp<volScalarField::Internal> blend
    (
        const volScalarField::Internal& F1,
        const dimensionedScalar& psi1,
        const dimensionedScalar& psi2
    ) const
    {
        return F1*(psi1 - psi2) + psi2;
    }

    tmp<volScalarField> alphaK(const volScalarField& F1) const
    {
        return blend(F1, alphaK1_, alphaK2_);
    }

    tmp<volScalarField> alphaOmega(const volScalarField& F1) const
    {
        return blend(F1, alphaOmega1_, alphaOmega2_);
    }

    tmp<volScalarField::Internal> beta
    (
        const volScalarField::Internal& F1
    ) const
    {
        return blend(F1, beta1_, beta2_);
    }

    tmp<volScalarField::Internal> gamma
    (
        const volScalarField::Internal& F1
    ) const
    {
        return blend(F1, gamma1_, gamma2_);
    }
    
    //- Return the effective diffusivity for k
    tmp<volScalarField> DkEff(const volScalarField& F1) const
    {
        return tmp<volScalarField>
        (
            new volScalarField("DkEff", alphaK(F1)*nut_ + this->getNu() )
        );
    }

    //- Return the effective diffusivity for omega
    tmp<volScalarField> DomegaEff(const volScalarField& F1) const
    {
        return tmp<volScalarField>
        (
            new volScalarField
            (
                "DomegaEff",
                alphaOmega(F1)*nut_ + this->getNu()
            )
        );
    }
    
    //- Return k production rate
    tmp<volScalarField::Internal> Pk
    (
        const volScalarField::Internal& G
    ) const;

    //- Return epsilon/k which for standard RAS is betaStar*omega
    tmp<volScalarField::Internal> epsilonByk
    (
        const volScalarField& F1,
        const volTensorField& gradU
    ) const;

    //- Return G/nu
    tmp<volScalarField::Internal> GbyNu
    (
        const volScalarField::Internal& GbyNu0,
        const volScalarField::Internal& F2,
        const volScalarField::Internal& S2
    ) const;

    tmp<fvScalarMatrix> kSource() const;

    tmp<fvScalarMatrix> omegaSource() const;
    
    tmp<fvScalarMatrix> Qsas
    (
        const volScalarField::Internal& S2,
        const volScalarField::Internal& gamma,
        const volScalarField::Internal& beta
    ) const;
    //@}

    /// \name Augmented variables for adjoint residuals
    //@{ 
    volScalarField& omega_;
    volScalarField omegaRes_;
    volScalarField omegaResRef_;
    volScalarField omegaResPartDeriv_;
    volScalarField omegaRef_;
    volScalarField& k_;
    volScalarField kRes_;
    volScalarField kResRef_;
    volScalarField kResPartDeriv_;
    volScalarField kRef_;
    //@}

    /// 3D wall distance
    const volScalarField& y_;

    /// cell-center omega values near the wall, this is to fix the issue that the 
    /// omegaWallFunction will try to modify omega values for the cells near walls
    /// this will cause issue for FD-based partial derivatives, so here we basically 
    /// implement a zeroGradient BC for near wall omega.
    scalarList omegaNearWall_;
 
public:

    TypeName ("kOmegaSST");
    // Constructors

    //- Construct from components
    AdjointkOmegaSST
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
    virtual ~AdjointkOmegaSST()
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

    /// save near wall omega values to omegaNearWall_    
    void saveOmegaNearWall();

    /// set omegaNearWall_ to near wall omega values
    void setOmegaNearWall();

    /// specially treatment to correct epsilon BC
    void correctOmegaBoundaryConditions();

    virtual void clearTurbVars();

    virtual void writeTurbStates();

};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
