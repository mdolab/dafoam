/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v5

\*---------------------------------------------------------------------------*/

#include "DAStateInfoInterFoam.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(DAStateInfoInterFoam, 0);
addToRunTimeSelectionTable(DAStateInfo, DAStateInfoInterFoam, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DAStateInfoInterFoam::DAStateInfoInterFoam(
    const word modelType,
    const fvMesh& mesh,
    const DAOption& daOption,
    const DAModel& daModel)
    : DAStateInfo(modelType, mesh, daOption, daModel)
{
    /*
    Description:
        Register the names of state variables
        NOTE:
        For model variables, such as turbulence model, register specific names
        For example, register "nut" to modelStates for RANS turbulence models,
        Then, we will call correctModelStates(stateInfo_["modelStates"]) to modify
        "nut" based on the selected turbulence model. For example, for SA model,
        correctModelStates will just replace "nut" with "nuTilda", for SST model,
        it will replace "nut" with "k" and append "omega" to modelStates.
        In other words, the model variables will be modified based on the selected
        models at runtime.
    */

    // read the alpha1 name
    IOdictionary transportProperties(
        IOobject(
            "transportProperties",
            mesh.time().constant(),
            mesh,
            IOobject::MUST_READ,
            IOobject::NO_WRITE,
            false));

    wordList phaseNames;
    transportProperties.readEntry<wordList>("phases", phaseNames);
    word alphaStateName = "alpha." + phaseNames[0];
    word alphaResidualName = "alpha." + phaseNames[0] + "Res";

    stateInfo_["volScalarStates"].append("p_rgh");
    stateInfo_["modelStates"].append("nut");
    stateInfo_["volVectorStates"].append("U");
    stateInfo_["volScalarStates"].append(alphaStateName);
    stateInfo_["surfaceScalarStates"].append("phi");

    // correct the names for model states based on the selected physical model at runtime
    daModel.correctModelStates(stateInfo_["modelStates"]);

    /* 
    Description:
        Adjoint state connectivity info, numbers denote the level of connectivity
        N/A means this state does not connect to the corrsponding residual 
    
                     U      T      p     nut    phi
        URes         2      2      1      1      0
        TRes         2      2      2      1      0
        pRes         3      2      2      2      1
        phiRes       2      2      1      1      0
    
        ******************************** NOTE 1 **********************************
        One does not need to specify connectivity for each physical model, set the 
        connectivity for original variables instead. For example, for turbulence models,
        set nut. Then, how is nut connected to the other turbulence states will be 
        set in the DAModel class. This is done by calling correctStateResidualModelCon. 
        For example, for SA model we just replace nut with nuTilda, for SST model, we need 
        to add extract connectivity since nut depends on grad(U), k, and omega. We need
        to do this for other pyhsical models such as radiation models.
        **************************************************************************
    
        ******************************** NOTE 2 **********************************
        Do not specify physical model connectivity here, because they will be added
        by calling addModelResidualCon. For example, for the SA turbulence
        model, it will add the nuTildaRes to stateResConInfo_ and setup
        its connectivity automatically.
        **************************************************************************

    */

    stateResConInfo_.set(
        "URes",
        {
            {"U", "p_rgh", "nut", alphaStateName, "phi"}, // lv0
            {"U", "p_rgh", alphaStateName, "nut"}, // lv1
            {"U", alphaStateName} // lv2
        });

    stateResConInfo_.set(
        "p_rghRes",
        {
            {"U", "p_rgh", "nut", alphaStateName, "phi"}, // lv0
            {"U", "p_rgh", "nut", alphaStateName, "phi"}, // lv1
            {"U", "p_rgh", "nut", alphaStateName}, // lv2
            {"U"} // lv3
        });

    stateResConInfo_.set(
        "phiRes",
        {
            {"U", "p_rgh", "nut", alphaStateName, "phi"}, // lv0
            {"U", "p_rgh", "nut", alphaStateName}, // lv1
            {"U", alphaStateName}, // lv2
        });

    stateResConInfo_.set(
        alphaResidualName,
        {
            {"U", "p_rgh", "nut", alphaStateName, "phi"}, // lv0
            {"U", "p_rgh", "nut", alphaStateName}, // lv1
            {"U", alphaStateName} // lv2
        });

    // need to correct connectivity for physical models for each residual
    daModel.correctStateResidualModelCon(stateResConInfo_["URes"]);
    daModel.correctStateResidualModelCon(stateResConInfo_["p_rghRes"]);
    daModel.correctStateResidualModelCon(stateResConInfo_["phiRes"]);
    daModel.correctStateResidualModelCon(stateResConInfo_[alphaResidualName]);

    // add physical model residual connectivity
    daModel.addModelResidualCon(stateResConInfo_);
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
