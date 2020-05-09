/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v1.0

    Description:
    Child class for the SpalartAllmaras model

\*---------------------------------------------------------------------------*/

#ifndef AdjointSpalartAllmaras_H
#define AdjointSpalartAllmaras_H

#include "AdjointRASModel.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                       Class AdjointSpalartAllmaras Declaration
\*---------------------------------------------------------------------------*/

class AdjointSpalartAllmaras
:
public AdjointRASModel
{

protected:

    /// \name SA parameters    
    //@{
    dimensionedScalar sigmaNut_;
    dimensionedScalar kappa_;
    dimensionedScalar Cb1_;
    dimensionedScalar Cb2_;
    dimensionedScalar Cw1_;
    dimensionedScalar Cw2_;
    dimensionedScalar Cw3_;
    dimensionedScalar Cv1_;
    dimensionedScalar Cs_;
    //@}
    
    /// \name SA functions
    //@{
    tmp<volScalarField> chi() const;

    tmp<volScalarField> fv1(const volScalarField& chi) const;

    tmp<volScalarField> fv2
    (
        const volScalarField& chi,
        const volScalarField& fv1
    ) const;

    tmp<volScalarField> Stilda
    (
        const volScalarField& chi,
        const volScalarField& fv1
    ) const;

    tmp<volScalarField> fw(const volScalarField& Stilda) const;
    
    tmp<volScalarField> DnuTildaEff() const;
    //@}

    /// \name Augmented variables for residual computation
    //@{
    volScalarField& nuTilda_;
    volScalarField nuTildaRes_;
    volScalarField nuTildaResRef_;
    volScalarField nuTildaResPartDeriv_;
    volScalarField nuTildaRef_;
    //@}

    /// 3D wall distance
    const volScalarField& y_;

public:

    TypeName ("SpalartAllmaras");
    // Constructors

    //- Construct from components
    AdjointSpalartAllmaras
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
    virtual ~AdjointSpalartAllmaras()
    {}
    
    
    // Member functions
    
    //virtual void perturbTurbStates
    //(
    //    const word& turbVar,
    //    const label cellI
    //);
    
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
