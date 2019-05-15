/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v1812

\*---------------------------------------------------------------------------*/

#include "AdjointDerivativeLaplacianFoam.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(AdjointDerivativeLaplacianFoam, 0);
addToRunTimeSelectionTable(AdjointDerivative, AdjointDerivativeLaplacianFoam, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

AdjointDerivativeLaplacianFoam::AdjointDerivativeLaplacianFoam
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
    setResidualClassMemberScalar(T,dimensionSet(0,0,-1,1,0,0,0))
{
    this->copyStates("Var2Ref"); // copy states to statesRef
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

void AdjointDerivativeLaplacianFoam::calcResiduals
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
        fvm::ddt(T_) - fvm::laplacian(DT, T_)
    );

    if(isRef) TResRef_  = TEqn&T_;
    else TRes_  = TEqn&T_;
    // need to normalize Res.
    normalizeResiduals(TRes);
    
    return;

}

void AdjointDerivativeLaplacianFoam::updateIntermediateVariables()
{
    // do nothing, we don't have intermediate state to update
}


} // End namespace Foam

// ************************************************************************* //