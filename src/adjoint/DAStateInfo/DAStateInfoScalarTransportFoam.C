/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v3

\*---------------------------------------------------------------------------*/

#include "DAStateInfoScalarTransportFoam.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(DAStateInfoScalarTransportFoam, 0);
addToRunTimeSelectionTable(DAStateInfo, DAStateInfoScalarTransportFoam, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DAStateInfoScalarTransportFoam::DAStateInfoScalarTransportFoam(
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

    stateInfo_["volScalarStates"].append("T");

    /* 
    Description:
        Adjoint state connectivity info, numbers denote the level of connectivity
        N/A means this state does not connect to the corrsponding residual 
    
                     D    
        DRes         2   
    
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
        "TRes",
        {
            {"T"}, // lv0
            {"T"}, // lv1
            {"T"} // lv2
        });
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
