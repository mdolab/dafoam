/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v1812

\*---------------------------------------------------------------------------*/

#include "AdjointJacobianConnectivitySimpleTFoam.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(AdjointJacobianConnectivitySimpleTFoam, 0);
addToRunTimeSelectionTable(AdjointJacobianConnectivity, AdjointJacobianConnectivitySimpleTFoam, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

AdjointJacobianConnectivitySimpleTFoam::AdjointJacobianConnectivitySimpleTFoam
(
    const fvMesh& mesh,
    const AdjointIO& adjIO,
    const AdjointSolverRegistry& adjReg,
    AdjointRASModel& adjRAS,
    AdjointIndexing& adjIdx
)
    :
    AdjointJacobianConnectivity(mesh,adjIO,adjReg,adjRAS,adjIdx)
{

    // Adj state connectivity levels for SimpleTFoam, numbers denote the level of connectivity
    // N/A means this state does not connect to the corrsponding residual 
    // ********************************NOTE**********************************
    // One does not need to specify connectivity for each turbulence model, set the connectivity
    // for nut only. How is nut connected to the other turbStates will be specified in the turbulence class
    // This is done by calling correctAdjStateResidualTurbCon. For example, for SA model we just replace
    // nut with nuTilda, for SST model, we need to add extract connectivity since nut depends on grad(U), k, and omega
    // **********************************************************************
    //              U      p     T     nut    phi
    // URes         2      1    N/A     1      0
    // pRes         3      2    N/A     2      1
    // TRes        N/A    N/A    2      1      0
    // phiRes       2      1    N/A     1      0
    
    adjStateResidualConInfo_.set
    (
        "URes",
        {
            {"U","p","nut","phi"}, // lv0
            {"U","p","nut"},       // lv1
            {"U"}                  // lv2
        }
    );
    
    adjStateResidualConInfo_.set
    (
        "pRes",
        {
            {"U","p","nut","phi"}, // lv0
            {"U","p","nut","phi"}, // lv1
            {"U","p","nut"},       // lv2
            {"U"}                  // lv3
        }
    );

    // NOTE: TEqn does not contain U but the alphatWallFunction does
    // so we need to add U to level0
    adjStateResidualConInfo_.set
    (
        "TRes",
        {
            {"T","nut","phi"}, // lv0
            {"T","nut"},       // lv1
            {"T"}              // lv2
        }
    );
    
    adjStateResidualConInfo_.set
    (
        "phiRes",
        {
            {"U","p","nut","phi"}, // lv0
            {"U","p","nut"},       // lv1
            {"U"},                 // lv2
        }
    );
    
    // need to correct turb con for each residual
    adjRAS.correctAdjStateResidualTurbCon(adjStateResidualConInfo_["URes"]);
    adjRAS.correctAdjStateResidualTurbCon(adjStateResidualConInfo_["pRes"]);
    adjRAS.correctAdjStateResidualTurbCon(adjStateResidualConInfo_["TRes"]);
    adjRAS.correctAdjStateResidualTurbCon(adjStateResidualConInfo_["phiRes"]);
    
    // add turbulence residual connectivity
    adjRAS.setAdjStateResidualTurbCon(adjStateResidualConInfo_);
    
    //Info<<adjStateResidualConInfo_<<endl;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
