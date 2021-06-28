/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v2

\*---------------------------------------------------------------------------*/

#ifdef CODI_AD_FORWARD

if (daOptionPtr_->getAllOptions().subDict("useAD").getWord("mode") == "forward")
{

    dictionary objFuncDict = daOptionPtr_->getAllOptions().subDict("objFunc");

    forAll(daObjFuncPtrList_, idxI)
    {
        DAObjFunc& daObjFunc = daObjFuncPtrList_[idxI];
        word objFuncName = daObjFunc.getObjFuncName();
        scalar objFunc = daObjFunc.getObjFuncValue();
        Info << endl << objFuncName
             << "-" << daObjFunc.getObjFuncPart()
             << "-" << daObjFunc.getObjFuncType()
             << ": Val " << objFunc << " Deriv " << objFunc.getGradient() << endl;
        
        // assign the forward mode AD derivative to forwardADDerivVal_
        // such that we can get this value later
        forwardADDerivVal_.set(objFuncName, objFunc.getGradient());
    }
}

Info << endl
     << "forwardModeAD done!" << endl;

#endif