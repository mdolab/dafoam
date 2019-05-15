/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v1812
    
\*---------------------------------------------------------------------------*/

#include "AdjointJacobianConnectivityBuoyantBoussinesqSimpleFoam.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(AdjointJacobianConnectivityBuoyantBoussinesqSimpleFoam, 0);
addToRunTimeSelectionTable(AdjointJacobianConnectivity, AdjointJacobianConnectivityBuoyantBoussinesqSimpleFoam, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

AdjointJacobianConnectivityBuoyantBoussinesqSimpleFoam::AdjointJacobianConnectivityBuoyantBoussinesqSimpleFoam
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
    // NOTE: phiRes connectivity level is based on its neighbour and owner cells, i.e., idxN and idxO
    // ********************************NOTE**********************************
    // One does not need to specify connectivity for each turbulence model, set the connectivity
    // for nut only. How is nut connected to the other turbStates will be specified in the turbulence class
    // This is done by calling correctAdjStateResidualTurbCon. For example, for SA model we just replace
    // nut with nuTilda, for SST model, we need to add extra connectivity since nut depends on grad(U), k, and omega
    // **********************************************************************
    //              U    p_rgh   T   G   nut    phi
    // URes         2      2     2  N/A   1      0
    // p_rghRes     3      2     2  N/A   2      1
    // TRes         0     N/A    2   0    1      0
    // GRes        N/A     N/A   0   2   N/A    N/A
    // phiRes       2      1     1  N/A   1      0
    
    adjStateResidualConInfo_.set
    (
        "URes",
        {
            {"U","p_rgh","T","nut","phi"}, // lv0
            {"U","p_rgh","T","nut"},       // lv1
            {"U","p_rgh","T"}              // lv2
        }
    );
    
    adjStateResidualConInfo_.set
    (
        "p_rghRes",
        {
            {"U","p_rgh","T","nut","phi"}, // lv0
            {"U","p_rgh","T","nut","phi"}, // lv1
            {"U","p_rgh","T","nut"},       // lv2
            {"U"}                          // lv3
        }
    );
    
    // NOTE: TEqn does not contain U but the alphatWallFunction does
    // so we need to add U to level0
    adjStateResidualConInfo_.set
    (
        "TRes",
        {
            {"U","T","nut","G","phi"}, // lv0
            {"T","nut"},       // lv1
            {"T"}              // lv2
        }
    );

    adjStateResidualConInfo_.set
    (
        "GRes",
        {
            {"G","T"},    // lv0
            {"G"},        // lv1
            {"G"}         // lv2
        }
    );
    
    adjStateResidualConInfo_.set
    (
        "phiRes",
        {
            {"U","p_rgh","T","nut","phi"}, // lv0
            {"U","p_rgh","T","nut"},       // lv1
            {"U"},                         // lv2
        }
    );
    
    // need to correct turb con for each residual
    adjRAS.correctAdjStateResidualTurbCon(adjStateResidualConInfo_["URes"]);
    adjRAS.correctAdjStateResidualTurbCon(adjStateResidualConInfo_["p_rghRes"]);
    adjRAS.correctAdjStateResidualTurbCon(adjStateResidualConInfo_["TRes"]);
    adjRAS.correctAdjStateResidualTurbCon(adjStateResidualConInfo_["GRes"]);
    adjRAS.correctAdjStateResidualTurbCon(adjStateResidualConInfo_["phiRes"]);
    
    // add turbulence residual connectivity
    adjRAS.setAdjStateResidualTurbCon(adjStateResidualConInfo_);
    
    //Info<<adjStateResidualConInfo_<<endl;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
