/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v2

\*---------------------------------------------------------------------------*/

#if defined(CODI_AD_REVERSE)

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

    DAObjFunc& daObjFunc = daObjFuncPtrList_[0];
    word objFuncName = daObjFunc.getObjFuncName();
    Info << objFuncName
         << "-" << daObjFunc.getObjFuncPart()
         << "-" << daObjFunc.getObjFuncType()
         << ": " << daObjFunc.getObjFuncValue() << endl;

    scalar objFunc = daObjFunc.getObjFuncValue();

    // register f output
    globalADTape_.registerOutput(objFunc);
    globalADTape_.setPassive();
    if (Pstream::master())
    {
        objFunc.setGradient(1.0);
    }
    globalADTape_.evaluate();

    if (type == "FFD")
    {
        Vec dFdXv;
        VecDuplicate(xvVec, &dFdXv);
        VecZeroEntries(dFdXv);
        forAll(meshPointsBFAD, i)
        {
            for (label j = 0; j < 3; j++)
            {
                label rowI = daIndexPtr_->getGlobalXvIndex(i, j);
                PetscScalar val = meshPointsBFAD[i][j].getGradient();
                VecSetValue(dFdXv, rowI, val, INSERT_VALUES);
            }
        }
        VecAssemblyBegin(dFdXv);
        VecAssemblyEnd(dFdXv);

        word outputName = "dFdXvBFAD_" + objFuncName;
        DAUtility::writeVectorBinary(dFdXv, outputName);
        DAUtility::writeVectorASCII(dFdXv, outputName);
    }
    else if (type == "Field")
    {
        word fieldName = dvSubDict.getWord("fieldName");
        word fieldType = dvSubDict.getWord("fieldType");

        label nComps = 0;
        if (fieldType == "vector")
        {
            nComps = 3;
        }
        else if (fieldType == "scalar")
        {
            nComps = 1;
        }

        Vec dFdField;
        VecCreate(PETSC_COMM_WORLD, &dFdField);
        VecSetSizes(dFdField, nComps * meshPtr_->nCells(), PETSC_DETERMINE);
        VecSetFromOptions(dFdField);
        VecZeroEntries(dFdField);
        this->assignFieldGradient2Vec(fieldName, fieldType, dFdField);

        word outputName = "dFdFieldBFAD_" + objFuncName;
        DAUtility::writeVectorBinary(dFdField, outputName);
        DAUtility::writeVectorASCII(dFdField, outputName);
    }
    else if (type == "ACTD")
    {
        Vec dFdACTD;
        VecCreate(PETSC_COMM_WORLD, &dFdACTD);
        VecSetSizes(dFdACTD, PETSC_DETERMINE, 9);
        VecSetFromOptions(dFdACTD);
        VecZeroEntries(dFdACTD);

        for (label i = 0; i < 9; i++)
        {
            PetscScalar valIn = actDVList[i].getGradient();
            // we need to do ADD_VALUES to get contribution from all procs
            VecSetValue(dFdACTD, i, valIn, ADD_VALUES);
        }

        VecAssemblyBegin(dFdACTD);
        VecAssemblyEnd(dFdACTD);

        word outputName = "dFdACTDBFAD_" + objFuncName;
        DAUtility::writeVectorBinary(dFdACTD, outputName);
        DAUtility::writeVectorASCII(dFdACTD, outputName);
    }

    globalADTape_.clearAdjoints();
    globalADTape_.reset();
}

Info << "BruteForceAD done!" << endl;

#endif