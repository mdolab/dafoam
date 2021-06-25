/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v2

\*---------------------------------------------------------------------------*/

#ifdef CODI_AD_FORWARD

Info << endl
     << "Forward-Mode AD is activated, setting the AD seeds..." << endl;
     
// for Xv derivs
pointField meshPointsForwardAD = mesh.points();
// for ACTD derivs
scalarList actDVListForwardAD(9);
if (daOptionPtr_->getAllOptions().subDict("useAD").getWord("mode") == "forward")
{

    dictionary designVarDict = daOptionPtr_->getAllOptions().subDict("designVar");

    if (designVarDict.size() != 1)
    {
        FatalErrorIn("")
            << "forward-mode AD only supports "
            << "one designVar key!"
            << abort(FatalError);
    }

    word designVarName = designVarDict.toc()[0];
    // get the subDict for this dvName
    dictionary dvSubDict = designVarDict.subDict(designVarName);
    word type = dvSubDict.getWord("designVarType");

    if (type == "FFD")
    {
        label seedIndex = daOptionPtr_->getAllOptions().subDict("useAD").getLabel("seedIndex");

        forAll(meshPointsForwardAD, i)
        {
            for (label j = 0; j < 3; j++)
            {
                label glxIdx = daIndexPtr_->getGlobalXvIndex(i, j);
                if (glxIdx == seedIndex || seedIndex == -1)
                {
                    meshPointsForwardAD[i][j].setGradient(1.0);
                }
            }
        }
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
                if (i == seedIndex || seedIndex == -1)
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
}

#endif