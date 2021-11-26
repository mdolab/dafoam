/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v2

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

    // Assign type, this is common for all objectives
    objFuncDict_.readEntry<word>("type", objFuncType_);
    varTypeFieldInversion_ = objFuncDict_.getWord("varTypeFieldInversion");
    stateType_ = objFuncDict_.getWord("stateType");
    stateName_ = objFuncDict_.getWord("stateName");
    stateRefName_ = objFuncDict_.getWord("stateRefName");
    scale_ = objFuncDict_.getScalar("scale");
    //weightedSum_ = objFuncDict_.getWord("weightedSum");
    objFuncDict_.readEntry<bool>("weightedSum", weightedSum_);
    if (varTypeFieldInversion_ == "surface")
    {
        objFuncDict_.readEntry<wordList>("patchNames", patchNames_);
    }
    if (stateType_ == "aeroCoeff")
    {
        scalarList dir; // decides if computing lift or drag
        objFuncDict_.readEntry<scalarList>("direction", dir);
        forceDir_[0] = dir[0];
        forceDir_[1] = dir[1];
        forceDir_[2] = dir[2];
        objFuncDict_.readEntry<scalar>("aeroCoeffRef", aeroCoeffRef_);
    }
    if (stateType_ == "surfacePressure")
    {
        objFuncDict_.readEntry<scalar>("pRef", pRef_);
    }
    if (weightedSum_ == True)
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
            if (weightedSum_ == True)
            {
                objFuncValue = weight_ * objFuncValue;
            }
            // need to reduce the sum of all objectives across all processors
            reduce(objFuncValue, sumOp<scalar>());
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
            if (weightedSum_ == True)
            {
                objFuncValue = weight_ * objFuncValue;
            }            

            // need to reduce the sum of all objectives across all processors
            reduce(objFuncValue, sumOp<scalar>());
        }
        /*else if (stateType_ == "ReynoldsShearStress")
        {
            const volSymmTensorField& stateRef = db.lookupObject<volSymmTensorField>(stateRefName_);

            volScalarField TauXYDNS(stateRef.component(symmTensor::XY)); 

            // read velocity and eddy viscosity field to compute the Reynold stresses
            const volVectorField& U_ = db.lookupObject<volVectorField>("U");
            const volScalarField& nut_ = db.lookupObject<volScalarField>("nut");

            // compute the Reynolds stress, assume ((2.0/3.0)*I)*tk() term is zero (true for S-A model)
            volSymmTensorField Tau(-(nut_)*dev(twoSymm(fvc::grad(U_))));
            // extract the XY component
            volScalarField TauXY(Tau.component(symmTensor::XY)); 

            forAll(objFuncCellSources, idxI)
            {
                const label& cellI = objFuncCellSources[idxI];
                objFuncCellValues[idxI] = (sqr(TauXY[cellI] - TauXYDNS[cellI]));
                objFuncValue += objFuncCellValues[idxI];
            }
            // need to reduce the sum of all objectives across all processors
            reduce(objFuncValue, sumOp<scalar>());

        }*/
    }

    else if (varTypeFieldInversion_ == "surface")
    {
        if (stateType_ == "surfaceFriction")
        {
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

                    objFuncValue += sqr(bSurfaceFriction - bSurfaceFrictionRef);
                }

                if (weightedSum_ == True)
                {
                    objFuncValue = weight_ * objFuncValue;
                }

            }

            // need to reduce the sum of all objectives across all processors
            reduce(objFuncValue, sumOp<scalar>());
        }

        else if (stateType_ == "aeroCoeff")
        {
            // get the ingredients for computations
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
            if (weightedSum_ == True)
            {
                objFuncValue = weight_ * objFuncValue;
            }
        }

        else if (stateType_ == "surfacePressure")
        {
            // get ref surface pressure "fields"
            const volScalarField& surfacePressureRef = db.lookupObject<volScalarField>(stateRefName_);

            // get the ingredient for computations
            const volScalarField& p = db.lookupObject<volScalarField>("p");

            forAll(patchNames_, cI)
            {
                label patchI = mesh_.boundaryMesh().findPatchID(patchNames_[cI]);
                const fvPatch& patch = mesh_.boundary()[patchI];
                forAll(patch, faceI)
                {

                    scalar bSurfacePressure = scale_ * (p.boundaryField()[patchI][faceI] - pRef_);

                    // calculate the objective function
                    // extract the reference surface pressure at the boundary
                    scalar bSurfacePressureRef = surfacePressureRef.boundaryField()[patchI][faceI];

                    objFuncValue += sqr(bSurfacePressure - bSurfacePressureRef);
                }
            }
            if (weightedSum_ == True)
            {
                objFuncValue = weight_ * objFuncValue;
            }

            // need to reduce the sum of all objectives across all processors
            reduce(objFuncValue, sumOp<scalar>());
        }
        else if (stateType_ == "surfacePressureCustom")
        {
            // get ref surface pressure "fields"
            const volScalarField& surfacePressureRef = db.lookupObject<volScalarField>(stateRefName_);

            // get the ingredient for computations
            const volScalarField& p = db.lookupObject<volScalarField>("p");

            forAll(patchNames_, cI)
            {
                label patchI = mesh_.boundaryMesh().findPatchID(patchNames_[cI]);
                const fvPatch& patch = mesh_.boundary()[patchI];
                forAll(patch, faceI)
                {

                    scalar bSurfacePressure = scale_ * (p.boundaryField()[patchI][faceI] - pRef_);

                    // calculate the objective function
                    // extract the reference surface pressure at the boundary
                    scalar bSurfacePressureRef = surfacePressureRef.boundaryField()[patchI][faceI];
                    if (bSurfacePressureRef < 1e16)
                    {
                        objFuncValue += sqr(bSurfacePressure - bSurfacePressureRef);
                    }
                }
            }

            // need to reduce the sum of all objectives across all processors
            reduce(objFuncValue, sumOp<scalar>());
        }
    }

    else if (varTypeFieldInversion_ == "profile")
    {
        // get the velocity field
        const volVectorField& state = db.lookupObject<volVectorField>(stateName_);

        // get the reference velocity field  (only x-component, hence volScalarField)
        const volScalarField& stateRef = db.lookupObject<volScalarField>(stateRefName_);

        forAll(objFuncCellSources, idxI)
        {
            const label& cellI = objFuncCellSources[idxI];
            if (stateRef[cellI] < 1e16)
            {
                objFuncCellValues[idxI] = (sqr(scale_ * state[cellI].x() - stateRef[cellI]));
                objFuncValue += objFuncCellValues[idxI];
            }
        }
        
        if (weightedSum_ == True)
        {
            objFuncValue = weight_ * objFuncValue;
        }

        // need to reduce the sum of all objectives across all processors
        reduce(objFuncValue, sumOp<scalar>());
    }
    return;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //