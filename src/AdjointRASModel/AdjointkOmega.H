/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v1812

    Description:
    Child class for the kOmega model

\*---------------------------------------------------------------------------*/

#ifndef AdjointkOmega_H
#define AdjointkOmega_H

#include "AdjointRASModel.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                       Class AdjointkOmega Declaration
\*---------------------------------------------------------------------------*/

class AdjointkOmega
:
public AdjointRASModel
{

protected:

    /// \name kOmega parameters    
    //@{
    dimensionedScalar Cmu_;
    dimensionedScalar beta_;
    dimensionedScalar gamma_;
    dimensionedScalar alphaK_;
    dimensionedScalar alphaOmega_;
    //@}
        
    /// \name kOmega functions   
    //@{

    tmp<volScalarField> DkEff() const
    {
        return tmp<volScalarField>
        (
            new volScalarField
            (
                "DkEff",
                alphaK_*nut_ + this->getNu()
            )
        );
    }

    tmp<volScalarField> DomegaEff() const
    {
        return tmp<volScalarField>
        (
            new volScalarField
            (
                "DomegaEff",
                alphaOmega_*nut_ + this->getNu()
            )
        );
    }
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

    /// cell-center omega values near the wall, this is to fix the issue that the 
    /// omegaWallFunction will try to modify omega values for the cells near walls
    /// this will cause issue for FD-based partial derivatives, so here we basically 
    /// implement a zeroGradient BC for near wall omega.
    scalarList omegaNearWall_;
 
public:

    TypeName ("kOmega");
    // Constructors

    //- Construct from components
    AdjointkOmega
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
    virtual ~AdjointkOmega()
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
