/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v2

\*---------------------------------------------------------------------------*/

#include "DAField.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DAField::DAField(
    const fvMesh& mesh,
    const DAOption& daOption,
    const DAModel& daModel,
    const DAIndex& daIndex)
    : mesh_(mesh),
      daOption_(daOption),
      daModel_(daModel),
      daIndex_(daIndex)
{
    // initialize stateInfo_
    word solverName = daOption.getOption<word>("solverName");
    autoPtr<DAStateInfo> daStateInfo(DAStateInfo::New(solverName, mesh, daOption, daModel));
    stateInfo_ = daStateInfo->getStateInfo();

    // check if we have special boundary conditions that need special treatment
    this->checkSpecialBCs();
}

void DAField::ofField2StateVec(Vec stateVec) const
{
    /*
    Description:
        Assign values for the state variable vector based on the 
        latest OpenFOAM field values

    Input:
        OpenFOAM field variables

    Output:
        stateVec: state variable vector

    Example:
        Image we have two state variables (p,T) and five cells, running on two CPU
        processors, the proc0 owns two cells and the proc1 owns three cells,
        then calling this function gives the state vector (state-by-state ordering):
    
        stateVec = [p0, p1, T0, T1 | p0, p1, p2, T0, T1, T2] <- p0 means p for the 0th cell on local processor
                     0   1   2   3 |  4   5   6   7   8   9  <- global state vec index
                   ---- proc0 -----|--------- proc1 ------- 
    */

    const objectRegistry& db = mesh_.thisDb();
    PetscScalar* stateVecArray;
    VecGetArray(stateVec, &stateVecArray);

    forAll(stateInfo_["volVectorStates"], idxI)
    {
        // lookup state from meshDb
        makeState(stateInfo_["volVectorStates"][idxI], volVectorField, db);

        forAll(mesh_.cells(), cellI)
        {
            for (label comp = 0; comp < 3; comp++)
            {
                label localIdx = daIndex_.getLocalAdjointStateIndex(stateName, cellI, comp);
                assignValueCheckAD(stateVecArray[localIdx], state[cellI][comp]);
            }
        }
    }

    forAll(stateInfo_["volScalarStates"], idxI)
    {
        // lookup state from meshDb
        makeState(stateInfo_["volScalarStates"][idxI], volScalarField, db);

        forAll(mesh_.cells(), cellI)
        {
            label localIdx = daIndex_.getLocalAdjointStateIndex(stateName, cellI);
            assignValueCheckAD(stateVecArray[localIdx], state[cellI]);
        }
    }

    forAll(stateInfo_["modelStates"], idxI)
    {
        // lookup state from meshDb
        makeState(stateInfo_["modelStates"][idxI], volScalarField, db);

        forAll(mesh_.cells(), cellI)
        {
            label localIdx = daIndex_.getLocalAdjointStateIndex(stateName, cellI);
            assignValueCheckAD(stateVecArray[localIdx], state[cellI]);
        }
    }

    forAll(stateInfo_["surfaceScalarStates"], idxI)
    {
        // lookup state from meshDb
        makeState(stateInfo_["surfaceScalarStates"][idxI], surfaceScalarField, db);

        forAll(mesh_.faces(), faceI)
        {
            label localIdx = daIndex_.getLocalAdjointStateIndex(stateName, faceI);
            if (faceI < daIndex_.nLocalInternalFaces)
            {
                assignValueCheckAD(stateVecArray[localIdx], state[faceI]);
            }
            else
            {
                label relIdx = faceI - daIndex_.nLocalInternalFaces;
                const label& patchIdx = daIndex_.bFacePatchI[relIdx];
                const label& faceIdx = daIndex_.bFaceFaceI[relIdx];
                assignValueCheckAD(stateVecArray[localIdx], state.boundaryField()[patchIdx][faceIdx]);
            }
        }
    }
    VecRestoreArray(stateVec, &stateVecArray);
}

void DAField::stateVec2OFField(const Vec stateVec) const
{
    /*
    Description:
        Assign values OpenFOAM field values based on the state variable vector

    Input:
    stateVec: state variable vector

    Output:
    OpenFoam field variables

    Example:
        Image we have two state variables (p,T) and five cells, running on two CPU
        processors, the proc0 owns two cells and the proc1 owns three cells,
        then calling this function will assign the p, and T based on the the state 
        vector (state-by-state ordering):
    
        stateVec = [p0, p1, T0, T1 | p0, p1, p2, T0, T1, T2] <- p0 means p for the 0th cell on local processor
                     0   1   2   3 |  4   5   6   7   8   9  <- global state vec index
                   ---- proc0 -----|--------- proc1 ------- 
    */

    const objectRegistry& db = mesh_.thisDb();
    const PetscScalar* stateVecArray;
    VecGetArrayRead(stateVec, &stateVecArray);

    forAll(stateInfo_["volVectorStates"], idxI)
    {
        // lookup state from meshDb
        makeState(stateInfo_["volVectorStates"][idxI], volVectorField, db);

        forAll(mesh_.cells(), cellI)
        {
            for (label comp = 0; comp < 3; comp++)
            {
                label localIdx = daIndex_.getLocalAdjointStateIndex(stateName, cellI, comp);
                state[cellI][comp] = stateVecArray[localIdx];
            }
        }
    }

    forAll(stateInfo_["volScalarStates"], idxI)
    {
        // lookup state from meshDb
        makeState(stateInfo_["volScalarStates"][idxI], volScalarField, db);

        forAll(mesh_.cells(), cellI)
        {
            label localIdx = daIndex_.getLocalAdjointStateIndex(stateName, cellI);
            state[cellI] = stateVecArray[localIdx];
        }
    }

    forAll(stateInfo_["modelStates"], idxI)
    {
        // lookup state from meshDb
        makeState(stateInfo_["modelStates"][idxI], volScalarField, db);

        forAll(mesh_.cells(), cellI)
        {
            label localIdx = daIndex_.getLocalAdjointStateIndex(stateName, cellI);
            state[cellI] = stateVecArray[localIdx];
        }
    }

    forAll(stateInfo_["surfaceScalarStates"], idxI)
    {
        // lookup state from meshDb
        makeState(stateInfo_["surfaceScalarStates"][idxI], surfaceScalarField, db);

        forAll(mesh_.faces(), faceI)
        {
            label localIdx = daIndex_.getLocalAdjointStateIndex(stateName, faceI);
            if (faceI < daIndex_.nLocalInternalFaces)
            {
                state[faceI] = stateVecArray[localIdx];
            }
            else
            {
                label relIdx = faceI - daIndex_.nLocalInternalFaces;
                const label& patchIdx = daIndex_.bFacePatchI[relIdx];
                const label& faceIdx = daIndex_.bFaceFaceI[relIdx];
                state.boundaryFieldRef()[patchIdx][faceIdx] = stateVecArray[localIdx];
            }
        }
    }
    VecRestoreArrayRead(stateVec, &stateVecArray);
}

void DAField::pointVec2OFMesh(const Vec xvVec) const
{
    /*
    Description:
        Assign the points in fvMesh of OpenFOAM based on the point vector

    Input:
        xvVec: a vector that stores the x, y, and z coordinates for all
        points in the fvMesh mesh

    Output:
        New mesh metrics in fvMesh, effectively by calling mesh.movePoints

    Example:
        Image we have three points in fvMesh, running on two CPU
        processors, the proc0 owns one point and the proc1 owns two points,
        then calling this function will assign xvVec based on the the points
        coordinates in fvMesh
    
        xvVec = [x0, y0, z0 | x0, y0, z0, x1, y1, z1] <- x0 means x coordinate for the 0th point on local processor
                 0   1   2  |  3   4   5   6   7   8  <- global point vec index
                --- proc0 --|--------- proc1 ------- 
    */

    const PetscScalar* xvVecArray;
    VecGetArrayRead(xvVec, &xvVecArray);

    pointField meshPoints(mesh_.points());

    forAll(mesh_.points(), pointI)
    {
        for (label comp = 0; comp < 3; comp++)
        {
            label localIdx = daIndex_.getLocalXvIndex(pointI, comp);
            meshPoints[pointI][comp] = xvVecArray[localIdx];
        }
    }

    VecRestoreArrayRead(xvVec, &xvVecArray);

    // movePoints update the mesh metrics such as volume, surface area and cell centers
    fvMesh& mesh = const_cast<fvMesh&>(mesh_);
    mesh.movePoints(meshPoints);
    mesh.moving(false);
}

void DAField::ofMesh2PointVec(Vec xvVec) const
{
    /*
    Description:
        Assign the point vector based on the points in fvMesh of OpenFOAM

    Input:
        Mesh coordinates in fvMesh

    Output:
        xvVec: a vector that stores the x, y, and z coordinates for all
        points in the fvMesh mesh

    Example:
        Image we have three points in fvMesh, running on two CPU
        processors, the proc0 owns one point and the proc1 owns two points,
        then calling this function will assign xvVec based on the the points
        coordinates in fvMesh
    
        xvVec = [x0, y0, z0 | x0, y0, z0, x1, y1, z1] <- x0 means x coordinate for the 0th point on local processor
                 0   1   2  |  3   4   5   6   7   8  <- global point vec index
                --- proc0 --|--------- proc1 ------- 
    */

    PetscScalar* xvVecArray;
    VecGetArray(xvVec, &xvVecArray);

    forAll(mesh_.points(), pointI)
    {
        for (label comp = 0; comp < 3; comp++)
        {
            label localIdx = daIndex_.getLocalXvIndex(pointI, comp);
            assignValueCheckAD(xvVecArray[localIdx], mesh_.points()[pointI][comp]);
        }
    }

    VecRestoreArray(xvVec, &xvVecArray);
}

void DAField::ofResField2ResVec(Vec resVec) const
{
    /*
    Description:
        Assign values for the residual vector based on the 
        latest OpenFOAM residual field values

    Input:
        OpenFOAM residual field variables

    Output:
        resVec: state residual  vector

    Example:
        Image we have two state residuals (pRes,TRes) and five cells, running on two CPU
        processors, the proc0 owns two cells and the proc1 owns three cells,
        then calling this function gives the residual vector (state-by-state ordering):
    
        resVec = [pRes0, pRes1, TRes0, TRes1 | pRes0, pRes1, pRes2, TRes0, TRes1, TRes2] 
                     0      1      2      3  |    4      5      6      7      8      9  <- global residual vec index
                   ---------- proc0 ---------|------------- proc1 ----------------------
        NOTE: pRes0 means p residual for the 0th cell on local processor
    */

    const objectRegistry& db = mesh_.thisDb();
    PetscScalar* stateResVecArray;
    VecGetArray(resVec, &stateResVecArray);

    forAll(stateInfo_["volVectorStates"], idxI)
    {
        // lookup state from meshDb
        makeStateRes(stateInfo_["volVectorStates"][idxI], volVectorField, db);

        forAll(mesh_.cells(), cellI)
        {
            for (label comp = 0; comp < 3; comp++)
            {
                label localIdx = daIndex_.getLocalAdjointStateIndex(stateName, cellI, comp);
                assignValueCheckAD(stateResVecArray[localIdx], stateRes[cellI][comp]);
            }
        }
    }

    forAll(stateInfo_["volScalarStates"], idxI)
    {
        // lookup state from meshDb
        makeStateRes(stateInfo_["volScalarStates"][idxI], volScalarField, db);

        forAll(mesh_.cells(), cellI)
        {
            label localIdx = daIndex_.getLocalAdjointStateIndex(stateName, cellI);
            assignValueCheckAD(stateResVecArray[localIdx], stateRes[cellI]);
        }
    }

    forAll(stateInfo_["modelStates"], idxI)
    {
        // lookup state from meshDb
        makeStateRes(stateInfo_["modelStates"][idxI], volScalarField, db);

        forAll(mesh_.cells(), cellI)
        {
            label localIdx = daIndex_.getLocalAdjointStateIndex(stateName, cellI);
            assignValueCheckAD(stateResVecArray[localIdx], stateRes[cellI]);
        }
    }

    forAll(stateInfo_["surfaceScalarStates"], idxI)
    {
        // lookup state from meshDb
        makeStateRes(stateInfo_["surfaceScalarStates"][idxI], surfaceScalarField, db);

        forAll(mesh_.faces(), faceI)
        {
            label localIdx = daIndex_.getLocalAdjointStateIndex(stateName, faceI);
            if (faceI < daIndex_.nLocalInternalFaces)
            {
                assignValueCheckAD(stateResVecArray[localIdx], stateRes[faceI]);
            }
            else
            {
                label relIdx = faceI - daIndex_.nLocalInternalFaces;
                const label& patchIdx = daIndex_.bFacePatchI[relIdx];
                const label& faceIdx = daIndex_.bFaceFaceI[relIdx];
                assignValueCheckAD(stateResVecArray[localIdx], stateRes.boundaryField()[patchIdx][faceIdx]);
            }
        }
    }
    VecRestoreArray(resVec, &stateResVecArray);
}

void DAField::resVec2OFResField(const Vec resVec) const
{
    /*
    Description:
        Assign OpenFOAM residual values based on the residual vector

    Input:
        resVec: residual vector

    Output:
        OpenFoam field variables

    Example:
        Image we have two state residuals (pRes,TRes) and five cells, running on two CPU
        processors, the proc0 owns two cells and the proc1 owns three cells,
        then calling this function gives the residual vector (state-by-state ordering):
    
        resVec = [pRes0, pRes1, TRes0, TRes1 | pRes0, pRes1, pRes2, TRes0, TRes1, TRes2] 
                     0      1      2      3  |    4      5      6      7      8      9  <- global residual vec index
                   ---------- proc0 ---------|------------- proc1 ----------------------
        NOTE: pRes0 means p residual for the 0th cell on local processor
    */

    const objectRegistry& db = mesh_.thisDb();
    const PetscScalar* stateResVecArray;
    VecGetArrayRead(resVec, &stateResVecArray);

    forAll(stateInfo_["volVectorStates"], idxI)
    {
        // lookup state from meshDb
        makeStateRes(stateInfo_["volVectorStates"][idxI], volVectorField, db);

        forAll(mesh_.cells(), cellI)
        {
            for (label comp = 0; comp < 3; comp++)
            {
                label localIdx = daIndex_.getLocalAdjointStateIndex(stateName, cellI, comp);
                stateRes[cellI][comp] = stateResVecArray[localIdx];
            }
        }
    }

    forAll(stateInfo_["volScalarStates"], idxI)
    {
        // lookup state from meshDb
        makeStateRes(stateInfo_["volScalarStates"][idxI], volScalarField, db);

        forAll(mesh_.cells(), cellI)
        {
            label localIdx = daIndex_.getLocalAdjointStateIndex(stateName, cellI);
            stateRes[cellI] = stateResVecArray[localIdx];
        }
    }

    forAll(stateInfo_["modelStates"], idxI)
    {
        // lookup state from meshDb
        makeStateRes(stateInfo_["modelStates"][idxI], volScalarField, db);

        forAll(mesh_.cells(), cellI)
        {
            label localIdx = daIndex_.getLocalAdjointStateIndex(stateName, cellI);
            stateRes[cellI] = stateResVecArray[localIdx];
        }
    }

    forAll(stateInfo_["surfaceScalarStates"], idxI)
    {
        // lookup state from meshDb
        makeStateRes(stateInfo_["surfaceScalarStates"][idxI], surfaceScalarField, db);

        forAll(mesh_.faces(), faceI)
        {
            label localIdx = daIndex_.getLocalAdjointStateIndex(stateName, faceI);
            if (faceI < daIndex_.nLocalInternalFaces)
            {
                stateRes[faceI] = stateResVecArray[localIdx];
            }
            else
            {
                label relIdx = faceI - daIndex_.nLocalInternalFaces;
                const label& patchIdx = daIndex_.bFacePatchI[relIdx];
                const label& faceIdx = daIndex_.bFaceFaceI[relIdx];
                stateRes.boundaryFieldRef()[patchIdx][faceIdx] = stateResVecArray[localIdx];
            }
        }
    }
    VecRestoreArrayRead(resVec, &stateResVecArray);
}

void DAField::checkSpecialBCs()
{
    /*
    Description:
        Check if we need to do special treatment for boundary conditions
        If any special BC is detected, append their names to specialBCs list
    */

    // *******************************************************************
    //                     pressureInletVelocity
    // *******************************************************************
    // Note we need to read the U field, instead of getting it from db
    // this is because coloringSolver does not read U
    // Also we need to set read_if_present for solid solvers in which
    // there is no U field
    volVectorField U(
        IOobject(
            "U",
            mesh_.time().timeName(),
            mesh_,
            IOobject::READ_IF_PRESENT,
            IOobject::NO_WRITE,
            false),
        mesh_,
        dimensionedVector("U", dimensionSet(0, 1, -1, 0, 0, 0, 0), vector::zero),
        zeroGradientFvPatchField<vector>::typeName);
    forAll(U.boundaryField(), patchI)
    {
        if (U.boundaryFieldRef()[patchI].type() == "pressureInletVelocity")
        {
            specialBCs.append("pressureInletVelocity");
            break;
        }
    }

    // *******************************************************************
    //                      append more special BCs
    // *******************************************************************
}

void DAField::specialBCTreatment()
{
    /*
    Description:
        Apply special treatment for boundary conditions
    */

    // *******************************************************************
    //                     pressureInletVelocity
    // *******************************************************************
    // for pressureInletVelocity, the inlet U depends on
    // rho and phi, so we need to call U.correctBoundaryConditions again
    if (DAUtility::isInList<word>("pressureInletVelocity", specialBCs))
    {

        volVectorField& U(const_cast<volVectorField&>(
            mesh_.thisDb().lookupObject<volVectorField>("U")));
        U.correctBoundaryConditions();
    }

    // *******************************************************************
    //                      append more special BCs
    // *******************************************************************
}

void DAField::setPrimalBoundaryConditions(const label printInfo)
{
    /*
    Description:
        A general function to read the inlet/outlet values from DAOption, and set
        the corresponding values to the boundary field. It also setup turbulence 
        wall boundary condition
        Note: this function should be called before running the primal solver
        If nothing is set, the BC will remain unchanged
        Example
        "primalBC":
        { 
            "bc0":
            {
                "patches": ["inlet", "side"], 
                "variable": "U", 
                "value": [10.0 0.0, 0.0],
            },
            "bc1":
            {
                "patches": ["outlet"], 
                "variable": "p", 
                "value": [101325.0],
            },
            useWallFunction True
        }
    */

    const objectRegistry& db = mesh_.thisDb();

    label setTurbWallBCs = 0;
    label useWallFunction = 0;

    const dictionary& allOptions = daOption_.getAllOptions();
    dictionary bcDict = allOptions.subDict("primalBC");

    forAll(bcDict.toc(), idxI)
    {
        word bcKey = bcDict.toc()[idxI];

        if (bcKey == "useWallFunction")
        {
            setTurbWallBCs = 1;
            useWallFunction = bcDict.getLabel("useWallFunction");
            continue;
        }

        dictionary bcSubDict = bcDict.subDict(bcKey);

        wordList patches;
        bcSubDict.readEntry<wordList>("patches", patches);
        word variable = bcSubDict.getWord("variable");
        scalarList value;
        bcSubDict.readEntry<scalarList>("value", value);

        // loop over all patches and set values
        forAll(patches, idxI)
        {
            word patch = patches[idxI];

            // it should be a scalar
            if (value.size() == 1)
            {
                if (!db.foundObject<volScalarField>(variable))
                {
                    if (printInfo)
                    {
                        Info << variable << " not found, skip it." << endl;
                    }
                    continue;
                }
                // it is a scalar
                volScalarField& state(const_cast<volScalarField&>(
                    db.lookupObject<volScalarField>(variable)));

                if (printInfo)
                {
                    Info << "Setting primal boundary conditions..." << endl;
                    Info << "Setting " << variable << " = " << value[0] << " at " << patch << endl;
                }

                label patchI = mesh_.boundaryMesh().findPatchID(patch);

                // for decomposed domain, don't set BC if the patch is empty
                if (mesh_.boundaryMesh()[patchI].size() > 0)
                {
                    if (state.boundaryFieldRef()[patchI].type() == "fixedValue")
                    {
                        forAll(state.boundaryFieldRef()[patchI], faceI)
                        {
                            state.boundaryFieldRef()[patchI][faceI] = value[0];
                        }
                    }
                    else if (state.boundaryFieldRef()[patchI].type() == "inletOutlet"
                             || state.boundaryFieldRef()[patchI].type() == "outletInlet")
                    {
                        // set value
                        forAll(state.boundaryFieldRef()[patchI], faceI)
                        {
                            state.boundaryFieldRef()[patchI][faceI] = value[0];
                        }
                        // set inletValue
                        mixedFvPatchField<scalar>& inletOutletPatch =
                            refCast<mixedFvPatchField<scalar>>(state.boundaryFieldRef()[patchI]);

                        inletOutletPatch.refValue() = value[0];
                    }
                    else if (state.boundaryFieldRef()[patchI].type() == "fixedGradient")
                    {
                        fixedGradientFvPatchField<scalar>& patchBC =
                            refCast<fixedGradientFvPatchField<scalar>>(state.boundaryFieldRef()[patchI]);
                        scalarField& grad = const_cast<scalarField&>(patchBC.gradient());
                        forAll(grad, idxI)
                        {
                            grad[idxI] = value[0];
                        }
                    }
                    else
                    {
                        FatalErrorIn("") << "only support fixedValues, inletOutlet, "
                                         << "outletInlet, fixedGradient!" << abort(FatalError);
                    }
                }
            }
            else if (value.size() == 3)
            {
                if (!db.foundObject<volVectorField>(variable))
                {
                    if (printInfo)
                    {
                        Info << variable << " not found, skip it." << endl;
                    }
                    continue;
                }
                // it is a vector
                volVectorField& state(const_cast<volVectorField&>(
                    db.lookupObject<volVectorField>(variable)));

                vector valVec = {value[0], value[1], value[2]};
                if (printInfo)
                {
                    Info << "Setting primal boundary conditions..." << endl;
                    Info << "Setting " << variable << " = (" << value[0] << " "
                         << value[1] << " " << value[2] << ") at " << patch << endl;
                }

                label patchI = mesh_.boundaryMesh().findPatchID(patch);

                // for decomposed domain, don't set BC if the patch is empty
                if (mesh_.boundaryMesh()[patchI].size() > 0)
                {
                    if (state.boundaryFieldRef()[patchI].type() == "fixedValue")
                    {
                        forAll(state.boundaryFieldRef()[patchI], faceI)
                        {
                            state.boundaryFieldRef()[patchI][faceI] = valVec;
                        }
                    }
                    else if (state.boundaryFieldRef()[patchI].type() == "inletOutlet"
                             || state.boundaryFieldRef()[patchI].type() == "outletInlet")
                    {
                        // set value
                        forAll(state.boundaryFieldRef()[patchI], faceI)
                        {
                            state.boundaryFieldRef()[patchI][faceI] = valVec;
                        }
                        // set inletValue
                        mixedFvPatchField<vector>& inletOutletPatch =
                            refCast<mixedFvPatchField<vector>>(state.boundaryFieldRef()[patchI]);
                        inletOutletPatch.refValue() = valVec;
                    }
                    else if (state.boundaryFieldRef()[patchI].type() == "fixedGradient")
                    {
                        fixedGradientFvPatchField<vector>& patchBC =
                            refCast<fixedGradientFvPatchField<vector>>(state.boundaryFieldRef()[patchI]);
                        vectorField& grad = const_cast<vectorField&>(patchBC.gradient());
                        forAll(grad, idxI)
                        {
                            grad[idxI][0] = value[0];
                            grad[idxI][1] = value[1];
                            grad[idxI][2] = value[2];
                        }
                    }
                    else
                    {
                        FatalErrorIn("") << "only support fixedValues, inletOutlet, "
                                         << "fixedGradient, and tractionDisplacement!"
                                         << abort(FatalError);
                    }
                }
            }
            else
            {
                FatalErrorIn("") << "value should be a list of either 1 (scalar) "
                                 << "or 3 (vector) elements" << abort(FatalError);
            }
        }
    }

    // we also set wall boundary conditions for turbulence variables
    if (setTurbWallBCs)
    {

        IOdictionary turbDict(
            IOobject(
                "turbulenceProperties",
                mesh_.time().constant(),
                mesh_,
                IOobject::MUST_READ,
                IOobject::NO_WRITE,
                false));

        dictionary coeffDict(turbDict.subDict("RAS"));
        word turbModelType = word(coeffDict["RASModel"]);
        //Info<<"turbModelType: "<<turbModelType<<endl;

        // create word regular expression for SA model
        wordReList SAModelWordReList {
            {"SpalartAllmaras.*", wordRe::REGEX}};
        wordRes SAModelWordRes(SAModelWordReList);

        // ------ nut ----------
        if (db.foundObject<volScalarField>("nut"))
        {

            volScalarField& nut(const_cast<volScalarField&>(
                db.lookupObject<volScalarField>("nut")));

            forAll(nut.boundaryField(), patchI)
            {
                if (mesh_.boundaryMesh()[patchI].type() == "wall")
                {
                    if (printInfo)
                    {
                        Info << "Setting nut wall BC for "
                             << mesh_.boundaryMesh()[patchI].name() << ". ";
                    }

                    if (useWallFunction)
                    {
                        // wall function for SA
                        if (SAModelWordRes(turbModelType))
                        {
                            nut.boundaryFieldRef().set(
                                patchI,
                                fvPatchField<scalar>::New(
                                    "nutUSpaldingWallFunction",
                                    mesh_.boundary()[patchI],
                                    nut));

                            if (printInfo)
                            {
                                Info << "BCType=nutUSpaldingWallFunction" << endl;
                            }
                        }
                        else // wall function for kOmega and kEpsilon
                        {
                            nut.boundaryFieldRef().set(
                                patchI,
                                fvPatchField<scalar>::New(
                                    "nutkWallFunction",
                                    mesh_.boundary()[patchI],
                                    nut));

                            if (printInfo)
                            {
                                Info << "BCType=nutkWallFunction" << endl;
                            }
                        }

                        // set boundary values
                        // for decomposed domain, don't set BC if the patch is empty
                        if (mesh_.boundaryMesh()[patchI].size() > 0)
                        {
                            scalar wallVal = nut[0];
                            forAll(nut.boundaryFieldRef()[patchI], faceI)
                            {
                                // assign uniform field
                                nut.boundaryFieldRef()[patchI][faceI] = wallVal;
                            }
                        }
                    }
                    else
                    {
                        nut.boundaryFieldRef().set(
                            patchI,
                            fvPatchField<scalar>::New(
                                "nutLowReWallFunction",
                                mesh_.boundary()[patchI],
                                nut));

                        if (printInfo)
                        {
                            Info << "BCType=nutLowReWallFunction" << endl;
                        }

                        // set boundary values
                        // for decomposed domain, don't set BC if the patch is empty
                        if (mesh_.boundaryMesh()[patchI].size() > 0)
                        {
                            forAll(nut.boundaryFieldRef()[patchI], faceI)
                            {
                                // assign uniform field
                                nut.boundaryFieldRef()[patchI][faceI] = 1e-14;
                            }
                        }
                    }
                }
            }
        }

        // ------ k ----------
        if (db.foundObject<volScalarField>("k"))
        {

            volScalarField& k(const_cast<volScalarField&>(
                db.lookupObject<volScalarField>("k")));

            forAll(k.boundaryField(), patchI)
            {
                if (mesh_.boundaryMesh()[patchI].type() == "wall")
                {
                    if (printInfo)
                    {
                        Info << "Setting k wall BC for "
                             << mesh_.boundaryMesh()[patchI].name() << ". ";
                    }

                    if (useWallFunction)
                    {
                        // wall function for SA
                        k.boundaryFieldRef().set(
                            patchI,
                            fvPatchField<scalar>::New("kqRWallFunction", mesh_.boundary()[patchI], k));

                        if (printInfo)
                        {
                            Info << "BCType=kqRWallFunction" << endl;
                        }

                        // set boundary values
                        // for decomposed domain, don't set BC if the patch is empty
                        if (mesh_.boundaryMesh()[patchI].size() > 0)
                        {
                            scalar wallVal = k[0];
                            forAll(k.boundaryFieldRef()[patchI], faceI)
                            {
                                k.boundaryFieldRef()[patchI][faceI] = wallVal; // assign uniform field
                            }
                        }
                    }
                    else
                    {
                        k.boundaryFieldRef().set(
                            patchI,
                            fvPatchField<scalar>::New("fixedValue", mesh_.boundary()[patchI], k));

                        if (printInfo)
                        {
                            Info << "BCType=fixedValue" << endl;
                        }

                        // set boundary values
                        // for decomposed domain, don't set BC if the patch is empty
                        if (mesh_.boundaryMesh()[patchI].size() > 0)
                        {
                            forAll(k.boundaryFieldRef()[patchI], faceI)
                            {
                                k.boundaryFieldRef()[patchI][faceI] = 1e-14; // assign uniform field
                            }
                        }
                    }
                }
            }
        }

        // ------ omega ----------
        if (db.foundObject<volScalarField>("omega"))
        {

            volScalarField& omega(const_cast<volScalarField&>(
                db.lookupObject<volScalarField>("omega")));

            forAll(omega.boundaryField(), patchI)
            {
                if (mesh_.boundaryMesh()[patchI].type() == "wall")
                {
                    if (printInfo)
                    {
                        Info << "Setting omega wall BC for "
                             << mesh_.boundaryMesh()[patchI].name() << ". ";
                    }

                    // always use omegaWallFunction
                    omega.boundaryFieldRef().set(
                        patchI,
                        fvPatchField<scalar>::New("omegaWallFunction", mesh_.boundary()[patchI], omega));

                    if (printInfo)
                    {
                        Info << "BCType=omegaWallFunction" << endl;
                    }

                    // set boundary values
                    // for decomposed domain, don't set BC if the patch is empty
                    if (mesh_.boundaryMesh()[patchI].size() > 0)
                    {
                        scalar wallVal = omega[0];
                        forAll(omega.boundaryFieldRef()[patchI], faceI)
                        {
                            omega.boundaryFieldRef()[patchI][faceI] = wallVal; // assign uniform field
                        }
                    }
                }
            }
        }

        // ------ epsilon ----------
        if (db.foundObject<volScalarField>("epsilon"))
        {

            volScalarField& epsilon(const_cast<volScalarField&>(
                db.lookupObject<volScalarField>("epsilon")));

            forAll(epsilon.boundaryField(), patchI)
            {
                if (mesh_.boundaryMesh()[patchI].type() == "wall")
                {

                    if (printInfo)
                    {
                        Info << "Setting epsilon wall BC for "
                             << mesh_.boundaryMesh()[patchI].name() << ". ";
                    }

                    if (useWallFunction)
                    {
                        epsilon.boundaryFieldRef().set(
                            patchI,
                            fvPatchField<scalar>::New("epsilonWallFunction", mesh_.boundary()[patchI], epsilon));

                        if (printInfo)
                        {
                            Info << "BCType=epsilonWallFunction" << endl;
                        }

                        // set boundary values
                        // for decomposed domain, don't set BC if the patch is empty
                        if (mesh_.boundaryMesh()[patchI].size() > 0)
                        {
                            scalar wallVal = epsilon[0];
                            forAll(epsilon.boundaryFieldRef()[patchI], faceI)
                            {
                                epsilon.boundaryFieldRef()[patchI][faceI] = wallVal; // assign uniform field
                            }
                        }
                    }
                    else
                    {
                        epsilon.boundaryFieldRef().set(
                            patchI,
                            fvPatchField<scalar>::New("fixedValue", mesh_.boundary()[patchI], epsilon));

                        if (printInfo)
                        {
                            Info << "BCType=fixedValue" << endl;
                        }

                        // set boundary values
                        // for decomposed domain, don't set BC if the patch is empty
                        if (mesh_.boundaryMesh()[patchI].size() > 0)
                        {
                            forAll(epsilon.boundaryFieldRef()[patchI], faceI)
                            {
                                epsilon.boundaryFieldRef()[patchI][faceI] = 1e-14; // assign uniform field
                            }
                        }
                    }
                }
            }
        }
    }
}

void DAField::ofField2List(
    scalarList& stateList,
    scalarList& stateBoundaryList) const
{
    /*
    Description:
        Assign values for the scalar list of states based on the latest OpenFOAM field values. 
        This function is similar to DAField::ofField2StateVec except that the output are 
        scalarLists instead of a Petsc vector

    Input:
        OpenFOAM field variables

    Output:
        stateList: scalar list of states
        stateBoundaryList: scalar list of boundary states

    Example:
        Image we have two state variables (p,T) and five cells, running on two CPU
        processors, the proc0 owns two cells and the proc1 owns three cells,
        then calling this function gives the scalar list of states (state-by-state ordering):
    
        scalarList = [p0, p1, T0, T1 | p0, p1, p2, T0, T1, T2] <- p0 means p for the 0th cell on local processor
                       0   1   2   3 |  4   5   6   7   8   9  <- global state index
                     ---- proc0 -----|--------- proc1 ------- 
    */

    const objectRegistry& db = mesh_.thisDb();

    label localBFaceI = 0;

    forAll(stateInfo_["volVectorStates"], idxI)
    {
        // lookup state from meshDb
        makeState(stateInfo_["volVectorStates"][idxI], volVectorField, db);

        forAll(mesh_.cells(), cellI)
        {
            for (label comp = 0; comp < 3; comp++)
            {
                label localIdx = daIndex_.getLocalAdjointStateIndex(stateName, cellI, comp);
                stateList[localIdx] = state[cellI][comp];
            }
        }

        forAll(state.boundaryField(), patchI)
        {
            if (state.boundaryField()[patchI].size() > 0)
            {
                forAll(state.boundaryField()[patchI], faceI)
                {
                    for (label comp = 0; comp < 3; comp++)
                    {
                        stateBoundaryList[localBFaceI] = state.boundaryField()[patchI][faceI][comp];
                        localBFaceI++;
                    }
                }
            }
        }
    }

    forAll(stateInfo_["volScalarStates"], idxI)
    {
        // lookup state from meshDb
        makeState(stateInfo_["volScalarStates"][idxI], volScalarField, db);

        forAll(mesh_.cells(), cellI)
        {
            label localIdx = daIndex_.getLocalAdjointStateIndex(stateName, cellI);
            stateList[localIdx] = state[cellI];
        }

        forAll(state.boundaryField(), patchI)
        {
            if (state.boundaryField()[patchI].size() > 0)
            {
                forAll(state.boundaryField()[patchI], faceI)
                {
                    stateBoundaryList[localBFaceI] = state.boundaryField()[patchI][faceI];
                    localBFaceI++;
                }
            }
        }
    }

    forAll(stateInfo_["modelStates"], idxI)
    {
        // lookup state from meshDb
        makeState(stateInfo_["modelStates"][idxI], volScalarField, db);

        forAll(mesh_.cells(), cellI)
        {
            label localIdx = daIndex_.getLocalAdjointStateIndex(stateName, cellI);
            stateList[localIdx] = state[cellI];
        }

        forAll(state.boundaryField(), patchI)
        {
            if (state.boundaryField()[patchI].size() > 0)
            {
                forAll(state.boundaryField()[patchI], faceI)
                {
                    stateBoundaryList[localBFaceI] = state.boundaryField()[patchI][faceI];
                    localBFaceI++;
                }
            }
        }
    }

    forAll(stateInfo_["surfaceScalarStates"], idxI)
    {
        // lookup state from meshDb
        makeState(stateInfo_["surfaceScalarStates"][idxI], surfaceScalarField, db);

        forAll(mesh_.faces(), faceI)
        {
            label localIdx = daIndex_.getLocalAdjointStateIndex(stateName, faceI);
            if (faceI < daIndex_.nLocalInternalFaces)
            {
                stateList[localIdx] = state[faceI];
            }
            else
            {
                label relIdx = faceI - daIndex_.nLocalInternalFaces;
                const label& patchIdx = daIndex_.bFacePatchI[relIdx];
                const label& faceIdx = daIndex_.bFaceFaceI[relIdx];
                stateList[localIdx] = state.boundaryField()[patchIdx][faceIdx];
            }
        }
    }
}

void DAField::list2OFField(
    const scalarList& stateList,
    const scalarList& stateBoundaryList,
    const label oldTimeLevel) const
{
    /*
    Description:
        Assign values OpenFOAM field values based on the scalar list of states
    
    Input:
    stateList: scalar list of states
    stateBoundaryList: scalar list of boundary states
    oldTimeLevel: assign to oldTime field instead of the original field, this will
      be used in time-accurate adjoint

    Output:
    OpenFoam field variables

    Example:
        Image we have two state variables (p,T) and five cells, running on two CPU
        processors, the proc0 owns two cells and the proc1 owns three cells,
        then calling this function gives the scalar list of states (state-by-state ordering):
    
        scalarList = [p0, p1, T0, T1 | p0, p1, p2, T0, T1, T2] <- p0 means p for the 0th cell on local processor
                       0   1   2   3 |  4   5   6   7   8   9  <- global state index
                     ---- proc0 -----|--------- proc1 ------- 
    */

    const objectRegistry& db = mesh_.thisDb();

    label localBFaceI = 0;

    if (oldTimeLevel > 2 || oldTimeLevel < 0)
    {
        FatalErrorIn("") << "oldTimeLevel not valid!"
                         << abort(FatalError);
    }

    forAll(stateInfo_["volVectorStates"], idxI)
    {
        // lookup state from meshDb
        makeState(stateInfo_["volVectorStates"][idxI], volVectorField, db);

        label maxOldTimes = state.nOldTimes();

        if (maxOldTimes >= oldTimeLevel)
        {
            forAll(mesh_.cells(), cellI)
            {
                for (label comp = 0; comp < 3; comp++)
                {
                    label localIdx = daIndex_.getLocalAdjointStateIndex(stateName, cellI, comp);
                    if (oldTimeLevel == 0)
                    {
                        state[cellI][comp] = stateList[localIdx];
                    }
                    else if (oldTimeLevel == 1)
                    {
                        state.oldTime()[cellI][comp] = stateList[localIdx];
                    }
                    else if (oldTimeLevel == 2)
                    {
                        state.oldTime().oldTime()[cellI][comp] = stateList[localIdx];
                    }
                }
            }

            forAll(state.boundaryField(), patchI)
            {
                if (state.boundaryField()[patchI].size() > 0)
                {
                    forAll(state.boundaryField()[patchI], faceI)
                    {
                        for (label comp = 0; comp < 3; comp++)
                        {
                            if (oldTimeLevel == 0)
                            {
                                state.boundaryFieldRef()[patchI][faceI][comp] =
                                    stateBoundaryList[localBFaceI];
                            }
                            else if (oldTimeLevel == 1)
                            {
                                state.oldTime().boundaryFieldRef()[patchI][faceI][comp] =
                                    stateBoundaryList[localBFaceI];
                            }
                            else if (oldTimeLevel == 2)
                            {
                                state.oldTime().oldTime().boundaryFieldRef()[patchI][faceI][comp] =
                                    stateBoundaryList[localBFaceI];
                            }
                            localBFaceI++;
                        }
                    }
                }
            }
        }
    }

    forAll(stateInfo_["volScalarStates"], idxI)
    {
        // lookup state from meshDb
        makeState(stateInfo_["volScalarStates"][idxI], volScalarField, db);

        label maxOldTimes = state.nOldTimes();

        if (maxOldTimes >= oldTimeLevel)
        {

            forAll(mesh_.cells(), cellI)
            {
                label localIdx = daIndex_.getLocalAdjointStateIndex(stateName, cellI);
                if (oldTimeLevel == 0)
                {
                    state[cellI] = stateList[localIdx];
                }
                else if (oldTimeLevel == 1)
                {
                    state.oldTime()[cellI] = stateList[localIdx];
                }
                else if (oldTimeLevel == 2)
                {
                    state.oldTime().oldTime()[cellI] = stateList[localIdx];
                }
            }

            forAll(state.boundaryField(), patchI)
            {
                if (state.boundaryField()[patchI].size() > 0)
                {
                    forAll(state.boundaryField()[patchI], faceI)
                    {
                        if (oldTimeLevel == 0)
                        {
                            state.boundaryFieldRef()[patchI][faceI] =
                                stateBoundaryList[localBFaceI];
                        }
                        else if (oldTimeLevel == 1)
                        {
                            state.oldTime().boundaryFieldRef()[patchI][faceI] =
                                stateBoundaryList[localBFaceI];
                        }
                        else if (oldTimeLevel == 2)
                        {
                            state.oldTime().oldTime().boundaryFieldRef()[patchI][faceI] =
                                stateBoundaryList[localBFaceI];
                        }
                        localBFaceI++;
                    }
                }
            }
        }
    }

    forAll(stateInfo_["modelStates"], idxI)
    {
        // lookup state from meshDb
        makeState(stateInfo_["modelStates"][idxI], volScalarField, db);

        label maxOldTimes = state.nOldTimes();

        if (maxOldTimes >= oldTimeLevel)
        {

            forAll(mesh_.cells(), cellI)
            {
                label localIdx = daIndex_.getLocalAdjointStateIndex(stateName, cellI);
                if (oldTimeLevel == 0)
                {
                    state[cellI] = stateList[localIdx];
                }
                else if (oldTimeLevel == 1)
                {
                    state.oldTime()[cellI] = stateList[localIdx];
                }
                else if (oldTimeLevel == 2)
                {
                    state.oldTime().oldTime()[cellI] = stateList[localIdx];
                }
            }

            forAll(state.boundaryField(), patchI)
            {
                if (state.boundaryField()[patchI].size() > 0)
                {
                    forAll(state.boundaryField()[patchI], faceI)
                    {
                        if (oldTimeLevel == 0)
                        {
                            state.boundaryFieldRef()[patchI][faceI] =
                                stateBoundaryList[localBFaceI];
                        }
                        else if (oldTimeLevel == 1)
                        {
                            state.oldTime().boundaryFieldRef()[patchI][faceI] =
                                stateBoundaryList[localBFaceI];
                        }
                        else if (oldTimeLevel == 2)
                        {
                            state.oldTime().oldTime().boundaryFieldRef()[patchI][faceI] =
                                stateBoundaryList[localBFaceI];
                        }
                        localBFaceI++;
                    }
                }
            }
        }
    }

    forAll(stateInfo_["surfaceScalarStates"], idxI)
    {
        // lookup state from meshDb
        makeState(stateInfo_["surfaceScalarStates"][idxI], surfaceScalarField, db);

        label maxOldTimes = state.nOldTimes();

        if (maxOldTimes >= oldTimeLevel)
        {

            forAll(mesh_.faces(), faceI)
            {
                label localIdx = daIndex_.getLocalAdjointStateIndex(stateName, faceI);
                if (faceI < daIndex_.nLocalInternalFaces)
                {

                    if (oldTimeLevel == 0)
                    {
                        state[faceI] = stateList[localIdx];
                    }
                    else if (oldTimeLevel == 1)
                    {
                        state.oldTime()[faceI] = stateList[localIdx];
                    }
                    else if (oldTimeLevel == 2)
                    {
                        state.oldTime().oldTime()[faceI] = stateList[localIdx];
                    }
                }
                else
                {
                    label relIdx = faceI - daIndex_.nLocalInternalFaces;
                    const label& patchIdx = daIndex_.bFacePatchI[relIdx];
                    const label& faceIdx = daIndex_.bFaceFaceI[relIdx];
                    if (oldTimeLevel == 0)
                    {
                        state.boundaryFieldRef()[patchIdx][faceIdx] =
                            stateList[localIdx];
                    }
                    else if (oldTimeLevel == 1)
                    {
                        state.oldTime().boundaryFieldRef()[patchIdx][faceIdx] =
                            stateList[localIdx];
                    }
                    else if (oldTimeLevel == 2)
                    {
                        state.oldTime().oldTime().boundaryFieldRef()[patchIdx][faceIdx] =
                            stateList[localIdx];
                    }
                }
            }
        }
    }
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
