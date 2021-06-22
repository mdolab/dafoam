/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v2

\*---------------------------------------------------------------------------*/

#if defined(CODI_AD_REVERSE)

Info << "BruteForceAD is activated, setting the AD seeds..." << endl;
// for Xv derivs
pointField meshPointsBFAD = mesh.points();
// for ACTD derivs
scalarList actDVList(9);
if (daOptionPtr_->getOption<label>("useBruteForceAD"))
{
    dictionary designVarDict = daOptionPtr_->getAllOptions().subDict("designVar");

    dictionary objFuncDict = daOptionPtr_->getAllOptions().subDict("objFunc");

    if (designVarDict.size() > 1 || objFuncDict.size() > 1)
    {
        FatalErrorIn("") << ""
                         << "bruteForceAD only supports "
                         << "one objFunc and one designVar!"
                         << abort(FatalError);
    }

    word designVarName = designVarDict.toc()[0];
    // get the subDict for this dvName
    dictionary dvSubDict = designVarDict.subDict(designVarName);
    word type = dvSubDict.getWord("designVarType");

    // setup AD inputs
    globalADTape_.reset();
    globalADTape_.setActive();

    if (type == "FFD")
    {
        forAll(meshPointsBFAD, i)
        {
            for (label j = 0; j < 3; j++)
            {
                globalADTape_.registerInput(meshPointsBFAD[i][j]);
            }
        }
        mesh.movePoints(meshPointsBFAD);
    }
    else if (type == "Field")
    {
        word fieldName = dvSubDict.getWord("fieldName");
        word fieldType = dvSubDict.getWord("fieldType");

        this->registerFieldVariableInput4AD(fieldName, fieldType);
        this->updateBoundaryConditions(fieldName, fieldType);
    }
    else if (type == "ACTD")
    {
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
                actDVList[i] = fvSource.getActuatorDVs(diskName, i);
            }

            // register  the input
            for (label i = 0; i < 9; i++)
            {
                this->globalADTape_.registerInput(actDVList[i]);
            }
            // set dv values to fvSource obj for all procs
            for (label i = 0; i < 9; i++)
            {
                fvSource.setActuatorDVs(diskName, i, actDVList[i]);
            }
            // the actuatorDVs are updated, now we need to recompute fvSource
            // this is not needed for the residual partials because fvSource
            // will be automatically calculated in the UEqn, but for the
            // obj partials, we need to manually recompute fvSource
            fvSource.updateFvSource();
        }
    }
}

#endif