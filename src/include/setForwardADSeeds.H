/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v3

\*---------------------------------------------------------------------------*/

#ifdef CODI_AD_FORWARD

Info << endl
     << "Forward-Mode AD is activated, setting the AD seeds..." << endl;

if (daOptionPtr_->getAllOptions().subDict("useAD").getWord("mode") == "forward")
{

    word designVarName = daOptionPtr_->getAllOptions().subDict("useAD").getWord("dvName");

    dictionary designVarDict = daOptionPtr_->getAllOptions().subDict("designVar");
    // get the subDict for this dvName
    dictionary dvSubDict = designVarDict.subDict(designVarName);
    word type = dvSubDict.getWord("designVarType");

    if (type == "FFD")
    {
        // NOTE: seedIndex is not needed for FFD here, instead, we set FFD2XvSeedVec
        // in pyDAFoam.py and then read it here
        // for Xv derivs
        pointField meshPointsForwardAD = mesh.points();

        Info << "Setting seeds based on FFD2XvSeedVec: " << endl;
        PetscScalar* seedArray;
        VecGetArray(FFD2XvSeedVec_, &seedArray);
        forAll(meshPointsForwardAD, i)
        {
            for (label j = 0; j < 3; j++)
            {
                label localIdx = daIndexPtr_->getLocalXvIndex(i, j);
                PetscScalar seedVal = seedArray[localIdx];
                meshPointsForwardAD[i][j].setGradient(seedVal);
            }
        }
        VecRestoreArray(FFD2XvSeedVec_, &seedArray);

        mesh.movePoints(meshPointsForwardAD);
    }
    else if (type == "Field")
    {
        label seedIndex = daOptionPtr_->getAllOptions().subDict("useAD").getLabel("seedIndex");

        word fieldName = dvSubDict.getWord("fieldName");
        word fieldType = dvSubDict.getWord("fieldType");

        if (fieldType == "scalar")
        {
            volScalarField& state = const_cast<volScalarField&>(
                meshPtr_->thisDb().lookupObject<volScalarField>(fieldName));

            forAll(state, cellI)
            {
                label glxIdx = daIndexPtr_->getGlobalCellIndex(cellI);
                if (glxIdx == seedIndex || seedIndex == -1)
                {
                    state[cellI].setGradient(1.0);
                }
            }
        }
        else if (fieldType == "vector")
        {
            volVectorField& state = const_cast<volVectorField&>(
                meshPtr_->thisDb().lookupObject<volVectorField>(fieldName));

            forAll(state, cellI)
            {
                for (label i = 0; i < 3; i++)
                {
                    label glxIdx = daIndexPtr_->getLocalCellVectorIndex(cellI, i);
                    if (glxIdx == seedIndex || seedIndex == -1)
                    {
                        state[cellI][i].setGradient(1.0);
                    }
                }
            }
        }
        else
        {
            FatalErrorIn("") << "fieldType not valid. Options: scalar or vector"
                             << abort(FatalError);
        }
    }
    else if (type == "ACTD")
    {
        // for ACTD derivs
        scalarList actDVListForwardAD(9);

        label seedIndex = daOptionPtr_->getAllOptions().subDict("useAD").getLabel("seedIndex");

        DAFvSource& fvSource = const_cast<DAFvSource&>(
            meshPtr_->thisDb().lookupObject<DAFvSource>("DAFvSource"));

        word diskName = dvSubDict.getWord("actuatorName");
        dictionary fvSourceSubDict = daOptionPtr_->getAllOptions().subDict("fvSource");
        word source = fvSourceSubDict.subDict(diskName).getWord("source");
        if (source == "cylinderAnnulusSmooth")
        {
            // get the design variable vals
            for (label i = 0; i < 9; i++)
            {
                if (i == seedIndex)
                {
                    actDVListForwardAD[i] = fvSource.getActuatorDVs(diskName, i);
                    actDVListForwardAD[i].setGradient(1.0);
                    fvSource.setActuatorDVs(diskName, i, actDVListForwardAD[i]);
                }
            }
            // the actuatorDVs are updated, now we need to recompute fvSource
            // this is not needed for the residual partials because fvSource
            // will be automatically calculated in the UEqn, but for the
            // obj partials, we need to manually recompute fvSource
            fvSource.updateFvSource();
        }
        else
        {
            FatalErrorIn("") << "only support cylinderAnnulusSmooth!"
                             << abort(FatalError);
        }
    }
    else if (type == "AOA")
    {
        // for AOA derivs
        scalar aoaForwardAD;

        // get info from dvSubDict. This needs to be defined in the pyDAFoam
        // name of the boundary patch
        wordList patches;
        dvSubDict.readEntry<wordList>("patches", patches);
        // the streamwise axis of aoa, aoa = tan( U_normal/U_flow )
        word flowAxis = dvSubDict.getWord("flowAxis");
        word normalAxis = dvSubDict.getWord("normalAxis");

        HashTable<label> axisIndices;
        axisIndices.set("x", 0);
        axisIndices.set("y", 1);
        axisIndices.set("z", 2);
        label flowAxisIndex = axisIndices[flowAxis];
        label normalAxisIndex = axisIndices[normalAxis];

        volVectorField& U = const_cast<volVectorField&>(
            meshPtr_->thisDb().lookupObject<volVectorField>("U"));

        // now we need to get the Ux, Uy values from the inout patches
        scalar Ux0 = -1e16, Uy0 = -1e16;
        forAll(patches, idxI)
        {
            word patchName = patches[idxI];
            label patchI = meshPtr_->boundaryMesh().findPatchID(patchName);
            if (meshPtr_->boundaryMesh()[patchI].size() > 0)
            {
                if (U.boundaryField()[patchI].type() == "fixedValue")
                {
                    Uy0 = U.boundaryField()[patchI][0][normalAxisIndex];
                    Ux0 = U.boundaryField()[patchI][0][flowAxisIndex];
                    break;
                }
                else if (U.boundaryField()[patchI].type() == "inletOutlet")
                {
                    mixedFvPatchField<vector>& inletOutletPatch =
                        refCast<mixedFvPatchField<vector>>(U.boundaryFieldRef()[patchI]);
                    Uy0 = inletOutletPatch.refValue()[0][normalAxisIndex];
                    Ux0 = inletOutletPatch.refValue()[0][flowAxisIndex];
                    break;
                }
                else
                {
                    FatalErrorIn("") << "boundaryType: " << U.boundaryFieldRef()[patchI].type()
                                     << " not supported!"
                                     << "Available options are: fixedValue, inletOutlet"
                                     << abort(FatalError);
                }
            }
        }
        // need to reduce the U value across all processors, this is because some of
        // the processors might not own the prescribed patches so their U value will be still -1e16, but
        // when calling the following reduce function, they will get the correct U from other processors
        reduce(Ux0, maxOp<scalar>());
        reduce(Uy0, maxOp<scalar>());
        aoaForwardAD = atan(Uy0 / Ux0);

        // convert to degree
        aoaForwardAD *= 180.0 / constant::mathematical::pi.getValue();

        Info << "Setting the seed for AOA: " << aoaForwardAD << endl;
        aoaForwardAD.setGradient(1.0);

        // convert to back to radian
        aoaForwardAD *= constant::mathematical::pi.getValue() / 180.0;

        // assign the angle of attack to the static variable in DAUtility. It will be used in determing
        // the forceDir in DAObjeFuncForce.C
        DAUtility::angleOfAttackRadForwardAD = aoaForwardAD;

        // set far field Ux, Uy
        forAll(patches, idxI)
        {
            word patchName = patches[idxI];
            label patchI = meshPtr_->boundaryMesh().findPatchID(patchName);

            if (meshPtr_->boundaryMesh()[patchI].size() > 0)
            {
                scalar UMag = sqrt(Ux0 * Ux0 + Uy0 * Uy0);
                scalar UxNew = UMag * cos(aoaForwardAD);
                scalar UyNew = UMag * sin(aoaForwardAD);

                if (U.boundaryField()[patchI].type() == "fixedValue")
                {
                    forAll(U.boundaryField()[patchI], faceI)
                    {
                        U.boundaryFieldRef()[patchI][faceI][flowAxisIndex] = UxNew;
                        U.boundaryFieldRef()[patchI][faceI][normalAxisIndex] = UyNew;
                    }
                }
                else if (U.boundaryField()[patchI].type() == "inletOutlet")
                {
                    mixedFvPatchField<vector>& inletOutletPatch =
                        refCast<mixedFvPatchField<vector>>(U.boundaryFieldRef()[patchI]);

                    forAll(U.boundaryField()[patchI], faceI)
                    {
                        inletOutletPatch.refValue()[faceI][flowAxisIndex] = UxNew;
                        inletOutletPatch.refValue()[faceI][normalAxisIndex] = UyNew;
                    }
                }
            }
        }
    }
    else if (type == "BC")
    {
        scalar BCForwardAD = -1e16;

        if (designVarName == "MRF")
        {
            const IOMRFZoneListDF& MRF = meshPtr_->thisDb().lookupObject<IOMRFZoneListDF>("MRFProperties");

            // first, we get the current value of omega and assign it to BC
            scalar& omega = const_cast<scalar&>(MRF.getOmegaRef());
            BCForwardAD = omega;

            Info << "Setting the seed for BCForwardAD: " << BCForwardAD << endl;
            BCForwardAD.setGradient(1.0);
            // now set BC to omega
            omega = BCForwardAD;
        }
        else
        {

            // get info from dvSubDict. This needs to be defined in the pyDAFoam
            // name of the variable for changing the boundary condition
            word varName = dvSubDict.getWord("variable");
            // name of the boundary patch
            wordList patches;
            dvSubDict.readEntry<wordList>("patches", patches);
            // the compoent of a vector variable, ignore when it is a scalar
            label comp = dvSubDict.getLabel("comp");

            // Now get the BC value
            forAll(patches, idxI)
            {
                word patchName = patches[idxI];
                label patchI = meshPtr_->boundaryMesh().findPatchID(patchName);
                if (meshPtr_->thisDb().foundObject<volVectorField>(varName))
                {
                    volVectorField& state(const_cast<volVectorField&>(
                        meshPtr_->thisDb().lookupObject<volVectorField>(varName)));
                    // for decomposed domain, don't set BC if the patch is empty
                    if (meshPtr_->boundaryMesh()[patchI].size() > 0)
                    {
                        if (state.boundaryFieldRef()[patchI].type() == "fixedValue")
                        {
                            BCForwardAD = state.boundaryFieldRef()[patchI][0][comp];
                        }
                        else if (state.boundaryFieldRef()[patchI].type() == "inletOutlet"
                                 || state.boundaryFieldRef()[patchI].type() == "outletInlet")
                        {
                            mixedFvPatchField<vector>& inletOutletPatch =
                                refCast<mixedFvPatchField<vector>>(state.boundaryFieldRef()[patchI]);
                            BCForwardAD = inletOutletPatch.refValue()[0][comp];
                        }
                    }
                }
                else if (meshPtr_->thisDb().foundObject<volScalarField>(varName))
                {
                    volScalarField& state(const_cast<volScalarField&>(
                        meshPtr_->thisDb().lookupObject<volScalarField>(varName)));
                    // for decomposed domain, don't set BC if the patch is empty
                    if (meshPtr_->boundaryMesh()[patchI].size() > 0)
                    {
                        if (state.boundaryFieldRef()[patchI].type() == "fixedValue")
                        {
                            BCForwardAD = state.boundaryFieldRef()[patchI][0];
                        }
                        else if (state.boundaryFieldRef()[patchI].type() == "inletOutlet"
                                 || state.boundaryFieldRef()[patchI].type() == "outletInlet")
                        {
                            mixedFvPatchField<scalar>& inletOutletPatch =
                                refCast<mixedFvPatchField<scalar>>(state.boundaryFieldRef()[patchI]);
                            BCForwardAD = inletOutletPatch.refValue()[0];
                        }
                    }
                }
            }
            // need to reduce the BC value across all processors, this is because some of
            // the processors might not own the prescribed patches so their BC value will be still -1e16, but
            // when calling the following reduce function, they will get the correct BC from other processors
            reduce(BCForwardAD, maxOp<scalar>());

            Info << "Setting the seed for BCForwardAD: " << BCForwardAD << endl;
            BCForwardAD.setGradient(1.0);

            // ******* now set BC ******
            forAll(patches, idxI)
            {
                word patchName = patches[idxI];
                label patchI = meshPtr_->boundaryMesh().findPatchID(patchName);
                if (meshPtr_->thisDb().foundObject<volVectorField>(varName))
                {
                    volVectorField& state(const_cast<volVectorField&>(
                        meshPtr_->thisDb().lookupObject<volVectorField>(varName)));
                    // for decomposed domain, don't set BC if the patch is empty
                    if (meshPtr_->boundaryMesh()[patchI].size() > 0)
                    {
                        if (state.boundaryFieldRef()[patchI].type() == "fixedValue")
                        {
                            forAll(state.boundaryFieldRef()[patchI], faceI)
                            {
                                state.boundaryFieldRef()[patchI][faceI][comp] = BCForwardAD;
                            }
                        }
                        else if (state.boundaryFieldRef()[patchI].type() == "inletOutlet"
                                 || state.boundaryFieldRef()[patchI].type() == "outletInlet")
                        {
                            mixedFvPatchField<vector>& inletOutletPatch =
                                refCast<mixedFvPatchField<vector>>(state.boundaryFieldRef()[patchI]);
                            vector val = inletOutletPatch.refValue()[0];
                            val[comp] = BCForwardAD;
                            inletOutletPatch.refValue() = val;
                        }
                    }
                }
                else if (meshPtr_->thisDb().foundObject<volScalarField>(varName))
                {
                    volScalarField& state(const_cast<volScalarField&>(
                        meshPtr_->thisDb().lookupObject<volScalarField>(varName)));
                    // for decomposed domain, don't set BC if the patch is empty
                    if (meshPtr_->boundaryMesh()[patchI].size() > 0)
                    {
                        if (state.boundaryFieldRef()[patchI].type() == "fixedValue")
                        {
                            forAll(state.boundaryFieldRef()[patchI], faceI)
                            {
                                state.boundaryFieldRef()[patchI][faceI] = BCForwardAD;
                            }
                        }
                        else if (state.boundaryFieldRef()[patchI].type() == "inletOutlet"
                                 || state.boundaryFieldRef()[patchI].type() == "outletInlet")
                        {
                            mixedFvPatchField<scalar>& inletOutletPatch =
                                refCast<mixedFvPatchField<scalar>>(state.boundaryFieldRef()[patchI]);
                            inletOutletPatch.refValue() = BCForwardAD;
                        }
                    }
                }
            }
        }
    }
}

#endif
