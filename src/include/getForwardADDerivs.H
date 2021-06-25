/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v2

\*---------------------------------------------------------------------------*/

#ifdef CODI_AD_FORWARD

if (daOptionPtr_->getAllOptions().subDict("forwardModeAD").getLabel("active"))
{

    dictionary objFuncDict = daOptionPtr_->getAllOptions().subDict("objFunc");

    dictionary designVarDict = daOptionPtr_->getAllOptions().subDict("designVar");

    if (designVarDict.size() != 1)
    {
        FatalErrorIn("")
            << "forward-mode AD only supports "
            << "one designVar key!"
            << abort(FatalError);
    }

    forAll(daObjFuncPtrList_, idxI)
    {
        DAObjFunc& daObjFunc = daObjFuncPtrList_[idxI];
        word objFuncName = daObjFunc.getObjFuncName();
        scalar objFunc = daObjFunc.getObjFuncValue();
        Info << objFuncName
             << "-" << daObjFunc.getObjFuncPart()
             << "-" << daObjFunc.getObjFuncType()
             << ": Val " << objFunc << " Deriv " << objFunc.getGradient() << endl;
    }
}

Info << endl
     << "forwardModeAD done!" << endl;

#endif