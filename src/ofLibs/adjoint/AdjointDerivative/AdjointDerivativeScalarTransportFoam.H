/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v1.0

    Description:
    Child class for scalarTransportFoam

\*---------------------------------------------------------------------------*/

#ifndef AdjointDerivativeScalarTransportFoam_H
#define AdjointDerivativeScalarTransportFoam_H

#include "AdjointDerivative.H"
#include "addToRunTimeSelectionTable.H"
#include "simpleControl.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                       Class AdjointDerivativeScalarTransportFoam Declaration
\*---------------------------------------------------------------------------*/

class AdjointDerivativeScalarTransportFoam
:
public AdjointDerivative
{

private:

protected:

    /// \name These are state variables, reference state variables, flow residuals, and partial derivatives
    //@{
    volScalarField& T_;
    volScalarField TRef_;
    volScalarField TRes_;
    volScalarField TResRef_;
    volScalarField TResPartDeriv_;
    surfaceScalarField& phi_;  
    //@}
    
public:

    TypeName ("scalarTransportDAFoam");
    // Constructors

    //- Construct from components
    AdjointDerivativeScalarTransportFoam
    (
        fvMesh& mesh,
        const AdjointIO& adjIO,
        const AdjointSolverRegistry& adjReg,
        AdjointRASModel& adjRAS,
        AdjointIndexing& adjIdx,
        AdjointJacobianConnectivity& adjCon,
        AdjointObjectiveFunction& adjObj
    );

    //- Destructor
    virtual ~AdjointDerivativeScalarTransportFoam()
    {}
    
    virtual void calcResiduals
    (
        const label isRef,
        const label isPC,
        const word fvMatrixName="None",
        const label updatePhi=0
    );
    
    virtual void updateIntermediateVariables();

    
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
