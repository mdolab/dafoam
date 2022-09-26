/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v3

\*---------------------------------------------------------------------------*/

#include "DAObjFuncFieldInversion.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(DAObjFuncFieldInversion, 0);
addToRunTimeSelectionTable(DAObjFunc, DAObjFuncFieldInversion, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DAObjFuncFieldInversion::DAObjFuncFieldInversion(
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

    objFuncDict_.readEntry<word>("type", objFuncType_);
    data_ = objFuncDict_.getWord("data");
    scale_ = objFuncDict_.getScalar("scale");
    objFuncDict_.readEntry<bool>("weightedSum", weightedSum_);
    if (weightedSum_ == true)
    {
        objFuncDict_.readEntry<scalar>("weight", weight_);
    }
    
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
void DAObjFuncFieldInversion::calcObjFunc(
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

    // initialize cell values to zero
    forAll(objFuncCellValues, idxI)
    {
        objFuncCellValues[idxI] = 0.0;
    }
    // initialize objFunValue
    objFuncValue = 0.0;

    const objectRegistry& db = mesh_.thisDb();

    if (data_ == "beta")
    {
        stateName_ = "betaFieldInversion";
        const volScalarField betaFieldInversion_ = db.lookupObject<volScalarField>(stateName_);
        
        forAll(objFuncCellSources, idxI)
        {
            const label& cellI = objFuncCellSources[idxI];
            objFuncCellValues[idxI] = scale_ * (sqr(betaFieldInversion_[cellI] - 1.0));
            objFuncValue += objFuncCellValues[idxI];
        }
        // need to reduce the sum of all objectives across all processors
        reduce(objFuncValue, sumOp<scalar>());

        if (weightedSum_ == true)
        {
            objFuncValue = weight_ * objFuncValue;
        }
    }
    else if (data_ == "UData")
    {
        stateName_ = "U";
        stateRefName_ = data_; 
        const volVectorField state = db.lookupObject<volVectorField>(stateName_);
        const volVectorField stateRef = db.lookupObject<volVectorField>(stateRefName_);
        forAll(objFuncCellSources, idxI)
        {
            const label& cellI = objFuncCellSources[idxI];
            scalar stateRefCell(mag(stateRef[cellI]));
            if (stateRefCell < 1e16)
            {
                objFuncCellValues[idxI] = (sqr(mag(scale_ * state[cellI] - stateRef[cellI])));
                objFuncValue += objFuncCellValues[idxI];
            }
        }

        // need to reduce the sum of all objectives across all processors
        reduce(objFuncValue, sumOp<scalar>());

        if (weightedSum_ == true)
        {
            objFuncValue = weight_ * objFuncValue;
        }
    }
    else if (data_ == "USingleComponentData")
    {
        stateName_ = "U";
        stateRefName_ = data_;
        scalarList velocityCompt;
        objFuncDict_.readEntry<scalarList>("velocityComponent", velocityCompt);
        vector velocityComponent_; 
        velocityComponent_[0] = velocityCompt[0];
        velocityComponent_[1] = velocityCompt[1];
        velocityComponent_[2] = velocityCompt[2];

        // get the velocity field
        const volVectorField state = db.lookupObject<volVectorField>(stateName_);

        // only use data for a specific component
        const volScalarField stateRef = db.lookupObject<volScalarField>(stateRefName_);

        forAll(objFuncCellSources, idxI)
        {
            const label& cellI = objFuncCellSources[idxI];
            if (stateRef[cellI] < 1e16)
            {
                vector URANS(state[cellI]);
                scalar UData(stateRef[cellI]); // assume only using one component of velocity
                objFuncCellValues[idxI] = (sqr(scale_ * (URANS & velocityComponent_) - UData));
                objFuncValue += objFuncCellValues[idxI];
            }
        }

        // need to reduce the sum of all objectives across all processors
        reduce(objFuncValue, sumOp<scalar>());

        if (weightedSum_ == true)
        {
            objFuncValue = weight_ * objFuncValue;
        }

    }
    else if (data_ == "pData")
    {
        stateName_ = "p";
        stateRefName_ = data_; 

        // get the pressure field
        const volScalarField& state = db.lookupObject<volScalarField>(stateName_);

        // get the reference pressure field
        const volScalarField& stateRef = db.lookupObject<volScalarField>(stateRefName_);

        forAll(objFuncCellSources, idxI)
        {
            const label& cellI = objFuncCellSources[idxI];
            if (stateRef[cellI] < 1e16)
            {
                objFuncCellValues[idxI] = (sqr(scale_ * state[cellI]- stateRef[cellI]));
                objFuncValue += objFuncCellValues[idxI];
            }
        }

        // need to reduce the sum of all objectives across all processors
        reduce(objFuncValue, sumOp<scalar>());

        if (weightedSum_ == true)
        {
            objFuncValue = weight_ * objFuncValue;
        }
    }
    else if (data_ == "surfacePressureData")
    {
        stateName_ = "p";
        stateRefName_ = "pData"; 

        const volScalarField surfacePressureRef = db.lookupObject<volScalarField>(stateRefName_);
        const volScalarField& p = db.lookupObject<volScalarField>(stateName_);

        wordList patchNames_; 
        objFuncDict_.readEntry<wordList>("patchNames", patchNames_);

        bool nonZeroPRefFlag_;
        objFuncDict_.readEntry<bool>("nonZeroPRef", nonZeroPRefFlag_);

        scalar pRef_(0.0);
        if (nonZeroPRefFlag_ == true)
        {
            scalarList pRefCoords;
            objFuncDict_.readEntry<scalarList>("pRefCoords", pRefCoords);
            vector pRefCoords_; 
            pRefCoords_[0] = pRefCoords[0];
            pRefCoords_[1] = pRefCoords[1];
            pRefCoords_[2] = pRefCoords[2];

            pRef_ = 0.0;
            label cellID = mesh_.findCell(pRefCoords_);
            // only assign pRef if the required cell is found in processor
            if (cellID != -1)
            {
                pRef_ = p[cellID];
            }
            reduce(pRef_, maxOp<scalar>());
        }
            
        forAll(patchNames_, cI)
        {
            label patchI = mesh_.boundaryMesh().findPatchID(patchNames_[cI]);
            const fvPatch& patch = mesh_.boundary()[patchI];
            forAll(patch, faceI)
            {
                scalar bSurfacePressure = scale_ * (p.boundaryField()[patchI][faceI] - pRef_);
                scalar bSurfacePressureRef = surfacePressureRef.boundaryField()[patchI][faceI];
                if (bSurfacePressureRef < 1e16)
                {
                    objFuncValue += sqr(bSurfacePressure - bSurfacePressureRef);
                }
            }
        }
        // need to reduce the sum of all objectives across all processors
        reduce(objFuncValue, sumOp<scalar>());
        if (weightedSum_ == true)
        {
            objFuncValue = weight_ * objFuncValue;
        }
    }
    else if(data_ == "surfaceFrictionData")
    {
        stateName_ = "surfaceFriction";
        stateRefName_ = data_;
        wordList patchNames_; 
        objFuncDict_.readEntry<wordList>("patchNames", patchNames_);

        scalarList dir;
        vector wssDir_; 
        objFuncDict_.readEntry<scalarList>("wssDir", dir);
        wssDir_[0] = dir[0];
        wssDir_[1] = dir[1];
        wssDir_[2] = dir[2];

        volScalarField& surfaceFriction = const_cast<volScalarField&>(db.lookupObject<volScalarField>(stateName_));
        const volScalarField surfaceFrictionRef = db.lookupObject<volScalarField>(stateRefName_);

        // ingredients for the computation
        tmp<volSymmTensorField> Reff = daTurb_.devRhoReff();
        volSymmTensorField::Boundary bReff = Reff().boundaryField();
        const surfaceVectorField::Boundary& Sfp = mesh_.Sf().boundaryField();
        const surfaceScalarField::Boundary& magSfp = mesh_.magSf().boundaryField();

        forAll(patchNames_, cI)
        {
            label patchI = mesh_.boundaryMesh().findPatchID(patchNames_[cI]);
            const fvPatch& patch = mesh_.boundary()[patchI];
            forAll(patch, faceI)
            {
                scalar bSurfaceFrictionRef = surfaceFrictionRef.boundaryField()[patchI][faceI];

                // normal vector at wall, use -ve sign to ensure vector pointing into the domain
                vector normal = -Sfp[patchI][faceI] / magSfp[patchI][faceI];

                // wall shear stress
                vector wss = normal & bReff[patchI][faceI];

                // wallShearStress or surfaceFriction (use surfaceFriction label to match the fields)
                scalar bSurfaceFriction = scale_ * (wss & wssDir_);

                surfaceFriction.boundaryFieldRef()[patchI][faceI] = bSurfaceFriction;

                // The following will allow to only use the Cf data at certain cells.
                // If you want to exclude cells, then given them a cell value of 1e16.
                if (bSurfaceFrictionRef < 1e16)
                {
                    // calculate the objective function
                    objFuncValue += sqr(bSurfaceFriction - bSurfaceFrictionRef);
                }
            }
        }

        // need to reduce the sum of all objectives across all processors
        reduce(objFuncValue, sumOp<scalar>());

        if (weightedSum_ == true)
        {
            objFuncValue = weight_ * objFuncValue;
        }
    }
    else if(data_ == "aeroCoeffData")
    {
        // get the ingredients for computations
        scalarList dir; // decides if computing lift or drag
        objFuncDict_.readEntry<scalarList>("direction", dir);
        vector forceDir_; 
        forceDir_[0] = dir[0];
        forceDir_[1] = dir[1];
        forceDir_[2] = dir[2];
        scalar aeroCoeffRef_; 
        objFuncDict_.readEntry<scalar>("aeroCoeffRef", aeroCoeffRef_);
        wordList patchNames_; 
        objFuncDict_.readEntry<wordList>("patchNames", patchNames_);

        const volScalarField& p = db.lookupObject<volScalarField>("p");
        const surfaceVectorField::Boundary& Sfb = mesh_.Sf().boundaryField();
        tmp<volSymmTensorField> tdevRhoReff = daTurb_.devRhoReff();
        const volSymmTensorField::Boundary& devRhoReffb = tdevRhoReff().boundaryField();

        vector forces(vector::zero);

        scalar aeroCoeff(0.0);

        forAll(patchNames_, cI)
        {
            // get the patch id label
            label patchI = mesh_.boundaryMesh().findPatchID(patchNames_[cI]);

            // create a shorter handle for the boundary patch
            const fvPatch& patch = mesh_.boundary()[patchI];

            // normal force
            vectorField fN(Sfb[patchI] * p.boundaryField()[patchI]);

            // tangential force
            vectorField fT(Sfb[patchI] & devRhoReffb[patchI]);

            forAll(patch, faceI)
            {
                forces.x() = fN[faceI].x() + fT[faceI].x();
                forces.y() = fN[faceI].y() + fT[faceI].y();
                forces.z() = fN[faceI].z() + fT[faceI].z();
                aeroCoeff += scale_ * (forces & forceDir_);
            }
        }
        // need to reduce the sum of all forces across all processors
        reduce(aeroCoeff, sumOp<scalar>());

        // compute the objective function
        objFuncValue += sqr(aeroCoeff - aeroCoeffRef_);

        // scale if performing weighted-sum multi-objective optimisation
        if (weightedSum_ == true)
        {
            objFuncValue = weight_ * objFuncValue;
        }
    }
    else if (data_ == "surfaceFrictionDataModified")
    {
        /* 
        In this modified implementation the wallShearStress computation different to the loop starting in line 283.
        Here the wallShearStress is computed using the equation shown in line 454. 
        (This modified equation is useful because this how the Cf is defined for the popular periodic hill case in
        literature.)
        */

        stateName_ = "surfaceFriction_";
        stateRefName_ = data_; 

        wordList patchNames_; 
        objFuncDict_.readEntry<wordList>("patchNames", patchNames_);

        // get surface friction "fields"
        volScalarField& surfaceFriction = const_cast<volScalarField&>(db.lookupObject<volScalarField>(stateName_));
        const volScalarField& surfaceFrictionRef = db.lookupObject<volScalarField>(stateRefName_);

        // ingredients for surface friction computation
        const volVectorField& U = db.lookupObject<volVectorField>("U");
        tmp<volTensorField> gradU = fvc::grad(U);
        const volTensorField::Boundary& bGradU = gradU().boundaryField();

        const surfaceVectorField::Boundary& Sfp = mesh_.Sf().boundaryField();
        const surfaceScalarField::Boundary& magSfp = mesh_.magSf().boundaryField();

        forAll(patchNames_, cI)
        {
            label patchI = mesh_.boundaryMesh().findPatchID(patchNames_[cI]);
            const fvPatch& patch = mesh_.boundary()[patchI];
            forAll(patch, faceI)
            {
                // normal vector at wall, use -ve sign to ensure vector pointing into the domain
                vector normal = -Sfp[patchI][faceI] / magSfp[patchI][faceI];

                // tangent vector, computed from normal: tangent = p x normal; where p is a unit vector perpidicular to the plane
                // NOTE: make this more general
                vector tangent(normal.y(), -normal.x(), 0.0);

                // velocity gradient at wall
                tensor fGradU = bGradU[patchI][faceI];

                /* need to get transpose of fGradU as it is stored in the following form:
                        fGradU = [ XX   XY  XZ  YX  YY  YZ  ZX  ZY  ZZ ]  (see slide 3 of: https://tinyurl.com/5ops2f5h)
                    but by definition,
                        grad(U) = [ XX  YX  ZX  XY  YY  ZY  XZ  YZ  ZZ]   
                    where XX = partial(u1)/partial(x1), YX = partial(u2)/partial(x1) and so on.
                        => grad(U) = transpose(fGradU)
                */
                tensor fGradUT = fGradU.T();

                /* compute the surface friction assuming incompressible flow and no wall functions! Add run time warning message 
                to reflect this later.
                        Cf = tau_wall/dynPressure,
                        tau_wall = rho * nu * tangent . (grad(U) . normal),
                        dynPressue = 0.5 * rho * mag(U_bulk^2),
                incorporate rho, mu, and dynamic pressure in scale_ as follows,
                        scale_ = nu / dynPressure,
                        => Cf = scale_ * tangent . (grad(U) . normal) */
                scalar bSurfaceFriction = scale_ * (tangent & (fGradUT & normal));

                surfaceFriction.boundaryFieldRef()[patchI][faceI] = bSurfaceFriction;

                // calculate the objective function
                // extract the reference surface friction at the boundary
                scalar bSurfaceFrictionRef = surfaceFrictionRef.boundaryField()[patchI][faceI];

                if (bSurfaceFrictionRef < 1e16)
                {
                    // calculate the objective function
                    objFuncValue += sqr(bSurfaceFriction - bSurfaceFrictionRef);
                }
            }
        }

        // need to reduce the sum of all objectives across all processors
        reduce(objFuncValue, sumOp<scalar>());

        if (weightedSum_ == true)
        {
            objFuncValue = weight_ * objFuncValue;
        }
    }
    else
    {
        FatalErrorIn("") << "dataType: " << data_
                            << " not supported for field inversion! "
                            << "Available options are: UData, pData, surfacePressureData, surfaceFrictionData, aeroCoeffData, and surfaceFrictionDataPeriodicHill."
                            << abort(FatalError);
    }

    return;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //