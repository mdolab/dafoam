/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v2

\*---------------------------------------------------------------------------*/

#include "DADummyTurbulenceModel.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(DADummyTurbulenceModel, 0);
addToRunTimeSelectionTable(DATurbulenceModel, DADummyTurbulenceModel, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DADummyTurbulenceModel::DADummyTurbulenceModel(
    const word modelType,
    const fvMesh& mesh,
    const DAOption& daOption)
    : DATurbulenceModel(modelType, mesh, daOption)
{
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
// Augmented functions
void DADummyTurbulenceModel::correctModelStates(wordList& modelStates) const
{
    /*
    Description:
        Update the name in modelStates based on the selected physical model at runtime

    Example:
        In DAStateInfo, if the modelStates reads:
        
        modelStates = {"nut"}
        
        then for the dummyTurbulencemodel, calling correctModelStates(modelStates) will give:
    
        modelStates={}

        We will remove nut from the list!
    */

    // remove nut from modelStates
    DAUtility::listDeleteVal<word>(modelStates, "nut");
}

void DADummyTurbulenceModel::correctNut()
{
    // Do nothing
}

void DADummyTurbulenceModel::correctBoundaryConditions()
{
    // Do nothing
}

void DADummyTurbulenceModel::updateIntermediateVariables()
{
    // Do nothing
}

void DADummyTurbulenceModel::correctStateResidualModelCon(List<List<word>>& stateCon) const
{
    /*
    Description:
        Update the original variable connectivity for the adjoint state 
        residuals in stateCon. Basically, we modify/add state variables based on the
        original model variables defined in stateCon.

    Input:
    
        stateResCon: the connectivity levels for a state residual, defined in Foam::DAJacCon

    Example:
        If stateCon reads:
        stateCon=
        {
            {"U", "p", "nut"},
            {"p"}
        }
    
        For the dummyTurbulenceModel, calling this function for will get a new stateCon
        stateCon=
        {
            {"U", "p"},
            {"p"}
        }
    */

    forAll(stateCon, idxI)
    {
        DAUtility::listDeleteVal<word>(stateCon[idxI], "nut");
    }
}

void DADummyTurbulenceModel::addModelResidualCon(HashTable<List<List<word>>>& allCon) const
{
    // Do nothing
}

void DADummyTurbulenceModel::correct()
{
    // Do nothing
}

void DADummyTurbulenceModel::calcResiduals(const dictionary& options)
{
    // Do nothing
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
