/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

\*---------------------------------------------------------------------------*/

#include "DAInputFieldUnsteady.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(DAInputFieldUnsteady, 0);
addToRunTimeSelectionTable(DAInput, DAInputFieldUnsteady, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DAInputFieldUnsteady::DAInputFieldUnsteady(
    const word inputName,
    const word inputType,
    fvMesh& mesh,
    const DAOption& daOption,
    const DAModel& daModel,
    const DAIndex& daIndex)
    : DAInput(
        inputName,
        inputType,
        mesh,
        daOption,
        daModel,
        daIndex)
{
    const dictionary& subDict = daOption_.getAllOptions().subDict("inputInfo").subDict(inputName);
    fieldName_ = subDict.getWord("fieldName");
    fieldType_ = subDict.getWord("fieldType");
    stepInterval_ = subDict.getLabel("stepInterval");
    interpolationMethod_ = subDict.getWord("interpolationMethod");

    scalar endTime = mesh.time().endTime().value();
    scalar deltaT = mesh.time().deltaT().value();
    label nSteps = round(endTime / deltaT);
    if (nSteps % stepInterval_ != 0)
    {
        FatalErrorIn("DAInputFieldUnsteady") << "Total number of time steps "
                                             << nSteps << "is not divisible by stepInterval "
                                             << stepInterval_ << exit(FatalError);
    }

    if (interpolationMethod_ == "linear")
    {
        // the parameter is the field at the interpolation point itself, here we include the field at t=0
        nParameters_ = nSteps / stepInterval_ + 1;
    }
    else if (interpolationMethod_ == "rbf")
    {
        // the parameters are weights and sigma for each interpolation point, so we have two parameter for each point at each step
        nParameters_ = 2 * (nSteps / stepInterval_ + 1);
    }
}

void DAInputFieldUnsteady::run(const scalarList& input)
{
    /*
    Description:
        Assign the input array to OF's field variables. Note that we need different treatment for distributed and 
        non-distributed field inputs

        For linear interpolation, the filed u are saved in this format

        ------ t = 0 ------|---- t=1interval ---|---- t=2interval ---
        u1, u2, u3, ... un | u1, u2, u3, ... un | u1, u2, u3, ... un

        For rbf interpolation, the data are saved in this format, here w and s are the parameters for rbf

       ------ t = 0 ------|---- t=1interval ---|---- t=2interval ---|------ t = 0 ------|---- t=1interval ---|---- t=2interval --
       w1, w2, w3, ... wn | w1, w2, w3, ... wn | w1, w2, w3, ... wn |s1, s2, s3, ... sn |s1, s2, s3, ... sn  | s1, s2, s3, ... sn|
    */

    if (input.size() != this->size())
    {
        FatalErrorIn("DAInputFieldUnsteady::run") << "the input size is not valid. " << exit(FatalError);
    }

    DAGlobalVar& globalVar =
        const_cast<DAGlobalVar&>(mesh_.thisDb().lookupObject<DAGlobalVar>("DAGlobalVar"));

    if (this->distributed())
    {
        forAll(globalVar.inputFieldUnsteady[inputName_], idxI)
        {
            globalVar.inputFieldUnsteady[inputName_][idxI] = input[idxI];
        }
    }
    else
    {
        // here input is a global array with multiple fields, while
        // globalVar.inputFieldUnsteady  is a local array with same number of fields
        if (fieldType_ == "scalar")
        {
            forAll(input, idxI)
            {
                label globalCellI = idxI % daIndex_.nGlobalCells;
                if (daIndex_.globalCellNumbering.isLocal(globalCellI))
                {
                    label localCellI = daIndex_.globalCellNumbering.toLocal(globalCellI);
                    label idxJ = idxI / daIndex_.nGlobalCells * daIndex_.nLocalCells + localCellI;
                    globalVar.inputFieldUnsteady[inputName_][idxJ] = input[idxI];
                }
            }
        }
        else
        {
            FatalErrorIn("DAInputFieldUnsteady::run") << "fieldType not valid" << exit(FatalError);
        }
    }
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
