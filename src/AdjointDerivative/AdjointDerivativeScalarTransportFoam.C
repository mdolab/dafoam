/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v1812

\*---------------------------------------------------------------------------*/

#include "AdjointDerivativeScalarTransportFoam.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(AdjointDerivativeScalarTransportFoam, 0);
addToRunTimeSelectionTable(AdjointDerivative, AdjointDerivativeScalarTransportFoam, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

AdjointDerivativeScalarTransportFoam::AdjointDerivativeScalarTransportFoam
(
    fvMesh& mesh,
    const AdjointIO& adjIO,
    const AdjointSolverRegistry& adjReg,
    AdjointRASModel& adjRAS,
    AdjointIndexing& adjIdx,
    AdjointJacobianConnectivity& adjCon,
    AdjointObjectiveFunction& adjObj
)
    :
    AdjointDerivative(mesh,adjIO,adjReg,adjRAS,adjIdx,adjCon,adjObj),
    // initialize and register state variables and their residuals, we use macros defined in macroFunctions.H
    setResidualClassMemberScalar(T,dimensionSet(0,0,-1,1,0,0,0)),
    phi_
    (
        const_cast<surfaceScalarField&>
        (
            db_.lookupObject<surfaceScalarField>("phi")
        )
    )
    
{
    this->copyStates("Var2Ref"); // copy states to statesRef
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

void AdjointDerivativeScalarTransportFoam::calcResiduals
(
    const label isRef,
    const label isPC,
    const word fvMatrixName,
    const label updatePhi
)
{
    dimensionedScalar DT
    (
        "DT1",
        dimArea/dimTime,
        adjIO_.flowProperties["DT"]
    );

    fvScalarMatrix TEqn
    (
        fvm::ddt(T_)
      + fvm::div(phi_, T_)
      - fvm::laplacian(DT, T_)
    );
    TEqn.relax();

    if(isRef) TResRef_  = TEqn&T_;
    else TRes_  = TEqn&T_;
    // need to normalize Res. 
    normalizeResiduals(TRes);
    
    return;

}

void AdjointDerivativeScalarTransportFoam::updateIntermediateVariables()
{
    // do nothing, we don't have intermediate state to update
}

} // End namespace Foam

// ************************************************************************* //