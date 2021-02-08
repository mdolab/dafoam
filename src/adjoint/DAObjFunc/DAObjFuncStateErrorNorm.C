/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v2

\*---------------------------------------------------------------------------*/

#include "DAObjFuncStateErrorNorm.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(DAObjFuncStateErrorNorm, 0);
addToRunTimeSelectionTable(DAObjFunc, DAObjFuncStateErrorNorm, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DAObjFuncStateErrorNorm::DAObjFuncStateErrorNorm(
    const fvMesh& mesh,
    const DAOption& daOption,
    const DAModel& daModel,
    const DAIndex& daIndex,
    const DAResidual& daResidual,
    const word objFuncName,
    const word objFuncPart,
    const dictionary& objFuncDict)
    : DAObjFunc(
        mesh,
        daOption,
        daModel,
        daIndex,
        daResidual,
        objFuncName,
        objFuncPart,
        objFuncDict),
    daTurb_(daModel.getDATurbulenceModel())

{

    // Assign type, this is common for all objectives
    objFuncDict_.readEntry<word>("type", objFuncType_);

    stateName_ = objFuncDict_.getWord("stateName");
    stateRefName_ = objFuncDict_.getWord("stateRefName");
    stateType_ = objFuncDict_.getWord("stateType");
    scale_ = objFuncDict_.getScalar("scale");
    varTypeFieldInversion_ = objFuncDict_.getWord("varTypeFieldInversion");
    objFuncDict_.readEntry<wordList>("patchNames", patchNames_); 

    // setup the connectivity, this is needed in Foam::DAJacCondFdW
    // this objFunc only depends on the state variable at the zero level cell
    if (DAUtility::isInList<word>(stateName_, daIndex.adjStateNames))
    {
        objFuncConInfo_ = {{stateName_}}; // level 0
    }
    else
    {
        objFuncConInfo_ = {{}}; // level 0
    }
}

/// calculate the value of objective function
void DAObjFuncStateErrorNorm::calcObjFunc(
    const labelList& objFuncFaceSources,
    const labelList& objFuncCellSources,
    scalarList& objFuncFaceValues,
    scalarList& objFuncCellValues,
    scalar& objFuncValue)
{
    /*
    Description:
        Calculate the stateErrorNorm
        f = scale * L2Norm( state-stateRef )

    Input:
        objFuncFaceSources: List of face source (index) for this objective
    
        objFuncCellSources: List of cell source (index) for this objective

    Output:
        objFuncFaceValues: the discrete value of objective for each face source (index). 
        This  will be used for computing df/dw in the adjoint.
    
        objFuncCellValues: the discrete value of objective on each cell source (index). 
        This will be used for computing df/dw in the adjoint.
    
        objFuncValue: the sum of objective, reduced across all processors and scaled by "scale"
    */

    // initialize to zero
    forAll(objFuncCellValues, idxI)
    {
        objFuncCellValues[idxI] = 0.0;
    }
    // initialize objFunValue
    objFuncValue = 0.0;

    const objectRegistry& db = mesh_.thisDb();

    if (varTypeFieldInversion_ == "volume")
    {
         if (stateType_ == "scalar")
        {
            const volScalarField& state = db.lookupObject<volScalarField>(stateName_);
            const volScalarField& stateRef = db.lookupObject<volScalarField>(stateRefName_);
            forAll(objFuncCellSources, idxI)
            {
                const label& cellI = objFuncCellSources[idxI];
                objFuncCellValues[idxI] = scale_ * (sqr(state[cellI] - stateRef[cellI]));
                objFuncValue += objFuncCellValues[idxI];
            }
        }
        else if (stateType_ == "vector")
        {
            const volVectorField& state = db.lookupObject<volVectorField>(stateName_);
            const volVectorField& stateRef = db.lookupObject<volVectorField>(stateRefName_);
            forAll(objFuncCellSources, idxI)
            {
                const label& cellI = objFuncCellSources[idxI];
                objFuncCellValues[idxI] = scale_ * (sqr(mag(state[cellI] - stateRef[cellI])));
                objFuncValue += objFuncCellValues[idxI];
            }
        }
    }
    
    else if (varTypeFieldInversion_ == "surface")
    {
         if (stateType_ == "surfaceFriction")
         {
             
            volScalarField surfaceFriction = db.lookupObject<volScalarField>(stateName_);

            const surfaceVectorField::Boundary& Sfp = mesh_.Sf().boundaryField();
	        const surfaceScalarField::Boundary& magSfp = mesh_.magSf().boundaryField();

	        tmp<volSymmTensorField> Reff = daTurb_.devRhoReff();   
	        const volSymmTensorField::Boundary& Reffp = Reff().boundaryField();
            
            forAll(patchNames_, cI)
            {
                label patchI = mesh_.boundaryMesh().findPatchID(patchNames_[cI]);
                const fvPatch& patch = mesh_.boundary()[patchI];
                forAll(patch,faceI)
                {
                    vector WSS = (-Sfp[patchI][faceI]/magSfp[patchI][faceI]) & Reffp[patchI][faceI];
                    // scale = 1 / (0.5 * rho * URef^2)
                    surfaceFriction.boundaryFieldRef()[patchI][faceI] = scale_ * mag(WSS);
                    //Info<<surfaceFriction.boundaryFieldRef()[patchI][faceI]<<endl;
                }
            }
             // compute the objective function 
            const volScalarField& surfaceFrictionRef = db.lookupObject<volScalarField>(stateRefName_);
            
            forAll(objFuncCellSources, idxI)  // at the moment surfaceFriction[cellI] and surfaceFrictionRef[cellI] are both zero
            {
                const label& cellI = objFuncCellSources[idxI];
               // Info<<surfaceFriction[cellI]<<endl; 
               // Info<<surfaceFrictionRef[cellI]<<endl; 
                objFuncCellValues[idxI] = sqr(surfaceFriction[cellI] - surfaceFrictionRef[cellI]); 
                objFuncValue += objFuncCellValues[idxI];
            }

         }
    }
    
    // need to reduce the sum of all objectives across all processors
    reduce(objFuncValue, sumOp<scalar>());

    return;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
