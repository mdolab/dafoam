/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v1.0

\*---------------------------------------------------------------------------*/

#include "AdjointJacobianConnectivityRhoSimpleFoam.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(AdjointJacobianConnectivityRhoSimpleFoam, 0);
addToRunTimeSelectionTable(AdjointJacobianConnectivity, AdjointJacobianConnectivityRhoSimpleFoam, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

AdjointJacobianConnectivityRhoSimpleFoam::AdjointJacobianConnectivityRhoSimpleFoam
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

    // Adj state connectivity levels for simpleFoam, numbers denote the level of connectivity
    // N/A means this state does not connect to the corrsponding residual 
    // ********************************NOTE**********************************
    // One does not need to specify connectivity for each turbulence model, set the connectivity
    // for nut only. How is nut connected to the other turbStates will be specified in the turbulence class
    // This is done by calling correctAdjStateResidualTurbCon. For example, for SA model we just replace
    // nut with nuTilda, for SST model, we need to add extract connectivity since nut depends on grad(U), k, and omega
    // **********************************************************************
    //              U      T      p     nut    phi
    // URes         2      2      1      1      0
    // TRes         2      2      2      1      0
    // pRes         3      2      2      2      1
    // phiRes       2      2      1      1      0
    
    adjStateResidualConInfo_.set
    (
        "URes",
        {
            {"U","p","T","nut","phi"}, // lv0
            {"U","p","T","nut"},       // lv1
            {"U","T"}                  // lv2
        }
    );

    adjStateResidualConInfo_.set
    (
        "TRes",
        {
            {"U","p","T","nut","phi"},  // lv0
            {"U","p","T","nut"},        // lv1
            {"U","p","T"}               // lv2
        }
    );
    
    adjStateResidualConInfo_.set
    (
        "pRes",
        {
            {"U","p","T","nut","phi"}, // lv0
            {"U","p","T","nut","phi"}, // lv1
            {"U","p","T","nut"},       // lv2
            {"U"}                      // lv3
        }
    );
    
    adjStateResidualConInfo_.set
    (
        "phiRes",
        {
            {"U","p","T","nut","phi"}, // lv0
            {"U","p","T","nut"},       // lv1
            {"U","T"},                 // lv2
        }
    );
    
    // need to correct turb con for each residual
    adjRAS.correctAdjStateResidualTurbCon(adjStateResidualConInfo_["URes"]);
    adjRAS.correctAdjStateResidualTurbCon(adjStateResidualConInfo_["TRes"]);
    adjRAS.correctAdjStateResidualTurbCon(adjStateResidualConInfo_["pRes"]);
    adjRAS.correctAdjStateResidualTurbCon(adjStateResidualConInfo_["phiRes"]);
    
    // add turbulence residual connectivity
    adjRAS.setAdjStateResidualTurbCon(adjStateResidualConInfo_);
    
    //Info<<adjStateResidualConInfo_<<endl;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
