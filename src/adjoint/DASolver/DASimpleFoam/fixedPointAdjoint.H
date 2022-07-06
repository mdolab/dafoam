/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v3

    Fixed-point adjoint members

\*---------------------------------------------------------------------------*/

// L2 norm of a scalar list (or the primitive field of a volScalarField)
// Scale off cell volumes
scalar DASimpleFoam::L2norm(const volScalarField& v)
{
    scalar L2norm_v = 0.0;

    forAll(v, cellI)
    {
        L2norm_v += sqr(v[cellI] / meshPtr_->V()[cellI]);
    }
    L2norm_v = sqrt(L2norm_v);

    return L2norm_v;
}

// Scale off cell volumes
vector DASimpleFoam::L2norm(const volVectorField& U)
{
    vector L2norm_U = vector::zero;

    forAll(U, cellI)
    {
        L2norm_U[0] += sqr(U[cellI][0] / meshPtr_->V()[cellI]);
        L2norm_U[1] += sqr(U[cellI][1] / meshPtr_->V()[cellI]);
        L2norm_U[2] += sqr(U[cellI][2] / meshPtr_->V()[cellI]);
    }
    L2norm_U[0] = sqrt(L2norm_U[0]);
    L2norm_U[1] = sqrt(L2norm_U[1]);
    L2norm_U[2] = sqrt(L2norm_U[2]);

    return L2norm_U;
}

// L2 norm of a surfaceScalarField
scalar DASimpleFoam::L2norm(const surfaceScalarField& Phi)
{
    scalar L2norm_Phi = 0.0;

    forAll(Phi.primitiveField(), cellI)
    {
        L2norm_Phi += sqr(Phi.primitiveField()[cellI]);
    }
    forAll(Phi.boundaryField(), patchI)
    {
        forAll(Phi.boundaryField()[patchI], faceI)
        {
            L2norm_Phi += sqr(Phi.boundaryField()[patchI][faceI]);
        }
    }
    L2norm_Phi = sqrt(L2norm_Phi);

    return L2norm_Phi;
}

label DASimpleFoam::runFPAdj(
    Vec dFdW,
    Vec psi)
{
#ifdef CODI_AD_REVERSE
    /*
    Description:
        Solve the adjoint using the fixed-point iteration method
    
    dFdW:
        The dF/dW vector 

    psi:
        The adjoint solution vector
    */

    VecZeroEntries(psi);

    word adjEqnSolMethod = daOptionPtr_->getOption<word>("adjEqnSolMethod");

    if (adjEqnSolMethod == "fixedPoint")
    {
        Info << "Solving the adjoint using fixed-point iteration method..." << endl;
        label fpMaxIters = daOptionPtr_->getSubDictOption<label>("adjEqnOption", "fpMaxIters");
        scalar relaxU = daOptionPtr_->getSubDictOption<scalar>("adjEqnOption", "relaxU");
        scalar relaxP = daOptionPtr_->getSubDictOption<scalar>("adjEqnOption", "relaxP");
        scalar relaxPhi = daOptionPtr_->getSubDictOption<scalar>("adjEqnOption", "relaxPhi");
        scalar relaxNuTilda = daOptionPtr_->getSubDictOption<scalar>("adjEqnOption", "relaxNuTilda");
        scalar fpRelTol = daOptionPtr_->getSubDictOption<scalar>("adjEqnOption", "fpRelTol");

        const objectRegistry& db = meshPtr_->thisDb();
        const volVectorField& U = db.lookupObject<volVectorField>("U");
        const volScalarField& p = db.lookupObject<volScalarField>("p");
        const surfaceScalarField& phi = db.lookupObject<surfaceScalarField>("phi");
        const volScalarField& nuTilda = db.lookupObject<volScalarField>("nuTilda");

        volVectorField dFdU("dFdU", 0.0 * U);
        volScalarField dFdP("dFdP", 0.0 * p);
        surfaceScalarField dFdPhi("dFdPhi", 0.0 * phi);
        volScalarField dFdNuTilda("dFdNuTilda", 0.0 * nuTilda);

        volVectorField URes("URes", 0.0 * U);
        volScalarField pRes("pRes", 0.0 * p);
        surfaceScalarField phiRes("phiRes", 0.0 * phi);
        volScalarField nuTildaRes("nuTildaRes", 0.0 * nuTilda);

        this->vec2Fields("vec2Field", dFdW, dFdU, dFdP, dFdPhi, dFdNuTilda);

        volVectorField UPsi("UPsi", 0.0 * U);
        volScalarField pPsi("pPsi", 0.0 * p);
        surfaceScalarField phiPsi("phiPsi", 0.0 * phi);
        volScalarField nuTildaPsi("nuTildaPsi", 0.0 * nuTilda);

        volVectorField pseudo_U("pseudo_U", U);
        volScalarField pseudo_p("pseudo_p", p);
        surfaceScalarField pseudo_phi("pseudo_phi", phi);
        volScalarField pseudo_nuTilda("pseudo_nuTilda", nuTilda);

        volVectorField adjURes("adjURes", 0.0 * U);
        volScalarField adjpRes("adjpRes", 0.0 * p);
        surfaceScalarField adjphiRes("adjphiRes", 0.0 * phi);
        volScalarField adjnuTildaRes("adjnuTildaRes", 0.0 * nuTilda);

        // Initiate the initial L2norm for adjURes, adjpRes, adjphiRes, adjnuTildaRes
        vector initNormAdjURes = vector::zero;
        scalar initNormAdjpRes = 0.0;
        scalar initNormAdjphiRes = 0.0;
        scalar initNormAdjnuTildaRes = 0.0;

        this->calcLduResiduals(URes, pRes, phiRes);
        daTurbulenceModelPtr_->calcLduResidualTurb(nuTildaRes);
        Info << "Residual for simpleFOAM after convergence: " << endl;
        Info << "L2 norm of URes: " << L2norm(URes) << endl;
        Info << "L2 norm of pRes: " << L2norm(pRes) << endl;
        Info << "L2 norm of phiRes: " << L2norm(phiRes) << endl;
        Info << "L2 norm of nuTildaRes: " << L2norm(nuTildaRes) << endl;

        label cnt = 0;
        while (cnt < fpMaxIters)
        {
            Info << "Step = " << cnt << "  Execution Time: " << meshPtr_->time().elapsedCpuTime() << " s" << endl;

            // ------ U -----
            this->calcAdjointResidual(URes, pRes, phiRes, nuTildaRes, dFdU, dFdP, dFdPhi, dFdNuTilda, UPsi, pPsi, phiPsi, nuTildaPsi, adjURes, adjpRes, adjphiRes, adjnuTildaRes, cnt);
            invTranProdUEqn(adjURes, pseudo_U);
            forAll(pseudo_U, cellI)
            {
                // the definition of adjURes has opposite sign than the formulation on Overleaf, so we do -= here
                UPsi[cellI] -= relaxU * pseudo_U[cellI];
            }

            //  ------- p ----
            this->calcAdjointResidual(URes, pRes, phiRes, nuTildaRes, dFdU, dFdP, dFdPhi, dFdNuTilda, UPsi, pPsi, phiPsi, nuTildaPsi, adjURes, adjpRes, adjphiRes, adjnuTildaRes, cnt);
            invTranProdPEqn(adjpRes, pseudo_p);
            forAll(pseudo_p, cellI)
            {
                pPsi[cellI] -= relaxP * pseudo_p[cellI];
            }

            // -------- phi  --------
            this->calcAdjointResidual(URes, pRes, phiRes, nuTildaRes, dFdU, dFdP, dFdPhi, dFdNuTilda, UPsi, pPsi, phiPsi, nuTildaPsi, adjURes, adjpRes, adjphiRes, adjnuTildaRes, cnt);

            forAll(adjphiRes, faceI)
            {
                phiPsi[faceI] += relaxPhi * adjphiRes[faceI];
            }
            forAll(adjphiRes.boundaryField(), patchI)
            {
                forAll(adjphiRes.boundaryField()[patchI], faceI)
                {
                    phiPsi.boundaryFieldRef()[patchI][faceI] += relaxPhi * adjphiRes.boundaryField()[patchI][faceI];
                }
            }

            // -------- nuTilda  --------
            this->calcAdjointResidual(URes, pRes, phiRes, nuTildaRes, dFdU, dFdP, dFdPhi, dFdNuTilda, UPsi, pPsi, phiPsi, nuTildaPsi, adjURes, adjpRes, adjphiRes, adjnuTildaRes, cnt);
            // Solve, using the inverse transpose product function
            daTurbulenceModelPtr_->invTranProdNuTildaEqn(adjnuTildaRes, pseudo_nuTilda);

            forAll(pseudo_nuTilda, cellI)
            {
                nuTildaPsi[cellI] -= relaxNuTilda * pseudo_nuTilda[cellI];
            }

            if (cnt >= 1)
            {
                vector norm_adjURes = L2norm(adjURes);
                scalar norm_adjpRes = L2norm(adjpRes);
                scalar norm_adjphiRes = L2norm(adjphiRes);
                scalar norm_adjnuTildaRes = L2norm(adjnuTildaRes);

                if (cnt == 1)
                {
                    initNormAdjURes = norm_adjURes;
                    initNormAdjpRes = norm_adjpRes;
                    initNormAdjphiRes = norm_adjphiRes;
                    initNormAdjnuTildaRes = norm_adjnuTildaRes;
                }

                // Normalize residual norms:
                norm_adjURes[0] /= initNormAdjURes[0];
                norm_adjURes[1] /= initNormAdjURes[1];
                norm_adjURes[2] /= initNormAdjURes[2];
                norm_adjpRes /= initNormAdjpRes;
                norm_adjphiRes /= initNormAdjphiRes;
                norm_adjnuTildaRes /= initNormAdjnuTildaRes;

                Info << "Normalized L2 norm of adjURes: " << norm_adjURes[0] << " " << norm_adjURes[1] << " " << norm_adjURes[2] << endl;
                Info << "Normalized L2 norm of adjpRes: " << norm_adjpRes << endl;
                Info << "Normalized L2 norm of adjphiRes: " << norm_adjphiRes << endl;
                Info << "Normalized L2 norm of adjnuTildaRes: " << norm_adjnuTildaRes << endl;

                // Check if fpRelTol is met:
                if (norm_adjURes[0] < fpRelTol && norm_adjURes[1] < fpRelTol && norm_adjURes[2] < fpRelTol && norm_adjpRes < fpRelTol && norm_adjphiRes < fpRelTol && norm_adjnuTildaRes < fpRelTol)
                {
                    Info << "Residual drop of " << fpRelTol << " has been achieved!" << endl;
                    break;
                }
            }

            cnt++;
        }

        // converged, assign the field Psi to psiVec
        this->vec2Fields("field2Vec", psi, UPsi, pPsi, phiPsi, nuTildaPsi);
    }
    else if (adjEqnSolMethod == "fixedPointC")
    {
        // not implemented yet
    }
    else
    {
        FatalErrorIn("adjEqnSolMethod not valid") << exit(FatalError);
    }

#endif
    return 0;
}

// Consistent fixed-point adjoint for SIMPLE
label DASimpleFoam::runCFPAdj(
    Vec dFdW,
    Vec psi)
{
#ifdef CODI_AD_REVERSE
    /*
    Description:
        Solve the adjoint using the fixed-point iteration method
    
    dFdW:
        The dF/dW vector 

    psi:
        The adjoint solution vector
    */

    VecZeroEntries(psi);

    word adjEqnSolMethod = daOptionPtr_->getOption<word>("adjEqnSolMethod");

    if (adjEqnSolMethod == "consistentFixedPoint")
    {
        Info << "Solving the adjoint using consistent fixed-point iteration method..." << endl;
        label fpMaxIters = daOptionPtr_->getSubDictOption<label>("adjEqnOption", "fpMaxIters");
        scalar relaxU = daOptionPtr_->getSubDictOption<scalar>("adjEqnOption", "relaxU");
        scalar relaxP = daOptionPtr_->getSubDictOption<scalar>("adjEqnOption", "relaxP");
        scalar relaxPhi = daOptionPtr_->getSubDictOption<scalar>("adjEqnOption", "relaxPhi");
        scalar relaxNuTilda = daOptionPtr_->getSubDictOption<scalar>("adjEqnOption", "relaxNuTilda");
        scalar fpRelTol = daOptionPtr_->getSubDictOption<scalar>("adjEqnOption", "fpRelTol");

        const objectRegistry& db = meshPtr_->thisDb();
        const volVectorField& U = db.lookupObject<volVectorField>("U");
        const volScalarField& p = db.lookupObject<volScalarField>("p");
        const surfaceScalarField& phi = db.lookupObject<surfaceScalarField>("phi");
        const volScalarField& nuTilda = db.lookupObject<volScalarField>("nuTilda");

        volVectorField dFdU("dFdU", 0.0 * U);
        volScalarField dFdP("dFdP", 0.0 * p);
        surfaceScalarField dFdPhi("dFdPhi", 0.0 * phi);
        volScalarField dFdNuTilda("dFdNuTilda", 0.0 * nuTilda);

        volVectorField URes("URes", 0.0 * U);
        volScalarField pRes("pRes", 0.0 * p);
        surfaceScalarField phiRes("phiRes", 0.0 * phi);
        volScalarField nuTildaRes("nuTildaRes", 0.0 * nuTilda);

        this->vec2Fields("vec2Field", dFdW, dFdU, dFdP, dFdPhi, dFdNuTilda);

        volVectorField UPsi("UPsi", 0.0 * U);
        volScalarField pPsi("pPsi", 0.0 * p);
        surfaceScalarField phiPsi("phiPsi", 0.0 * phi);
        volScalarField nuTildaPsi("nuTildaPsi", 0.0 * nuTilda);

        volVectorField pseudo_U("pseudo_U", U);
        volScalarField pseudo_p("pseudo_p", p);
        surfaceScalarField pseudo_phi("pseudo_phi", phi);
        volScalarField pseudo_nuTilda("pseudo_nuTilda", nuTilda);

        volVectorField adjURes("adjURes", 0.0 * U);
        volScalarField adjpRes("adjpRes", 0.0 * p);
        surfaceScalarField adjphiRes("adjphiRes", 0.0 * phi);
        volScalarField adjnuTildaRes("adjnuTildaRes", 0.0 * nuTilda);

        // Initiate the initial L2norm for adjURes, adjpRes, adjphiRes, adjnuTildaRes
        vector initNormAdjURes = vector::zero;
        scalar initNormAdjpRes = 0.0;
        scalar initNormAdjphiRes = 0.0;
        scalar initNormAdjnuTildaRes = 0.0;

        this->calcLduResiduals(URes, pRes, phiRes);
        daTurbulenceModelPtr_->calcLduResidualTurb(nuTildaRes);
        Info << "Residual for simpleFOAM after convergence: " << endl;
        Info << "L2 norm of URes: " << L2norm(URes) << endl;
        Info << "L2 norm of pRes: " << L2norm(pRes) << endl;
        Info << "L2 norm of phiRes: " << L2norm(phiRes) << endl;
        Info << "L2 norm of nuTildaRes: " << L2norm(nuTildaRes) << endl;

        label cnt = 0;
        while (cnt < fpMaxIters)
        {
            Info << "Step = " << cnt << "  Execution Time: " << meshPtr_->time().elapsedCpuTime() << " s" << endl;

            // ------ U -----
            this->calcAdjointResidual(URes, pRes, phiRes, nuTildaRes, dFdU, dFdP, dFdPhi, dFdNuTilda, UPsi, pPsi, phiPsi, nuTildaPsi, adjURes, adjpRes, adjphiRes, adjnuTildaRes, cnt);
            invTranProdUEqn(adjURes, pseudo_U);
            forAll(pseudo_U, cellI)
            {
                // the definition of adjURes has opposite sign than the formulation on Overleaf, so we do -= here
                UPsi[cellI] -= relaxU * pseudo_U[cellI];
            }

            //  ------- p ----
            this->calcAdjointResidual(URes, pRes, phiRes, nuTildaRes, dFdU, dFdP, dFdPhi, dFdNuTilda, UPsi, pPsi, phiPsi, nuTildaPsi, adjURes, adjpRes, adjphiRes, adjnuTildaRes, cnt);
            invTranProdPEqn(adjpRes, pseudo_p);
            forAll(pseudo_p, cellI)
            {
                pPsi[cellI] -= relaxP * pseudo_p[cellI];
            }

            // -------- phi  --------
            this->calcAdjointResidual(URes, pRes, phiRes, nuTildaRes, dFdU, dFdP, dFdPhi, dFdNuTilda, UPsi, pPsi, phiPsi, nuTildaPsi, adjURes, adjpRes, adjphiRes, adjnuTildaRes, cnt);

            forAll(adjphiRes, faceI)
            {
                phiPsi[faceI] += relaxPhi * adjphiRes[faceI];
            }
            forAll(adjphiRes.boundaryField(), patchI)
            {
                forAll(adjphiRes.boundaryField()[patchI], faceI)
                {
                    phiPsi.boundaryFieldRef()[patchI][faceI] += relaxPhi * adjphiRes.boundaryField()[patchI][faceI];
                }
            }

            // -------- nuTilda  --------
            this->calcAdjointResidual(URes, pRes, phiRes, nuTildaRes, dFdU, dFdP, dFdPhi, dFdNuTilda, UPsi, pPsi, phiPsi, nuTildaPsi, adjURes, adjpRes, adjphiRes, adjnuTildaRes, cnt);
            // Solve, using the inverse transpose product function
            daTurbulenceModelPtr_->invTranProdNuTildaEqn(adjnuTildaRes, pseudo_nuTilda);

            forAll(pseudo_nuTilda, cellI)
            {
                nuTildaPsi[cellI] -= relaxNuTilda * pseudo_nuTilda[cellI];
            }

            if (cnt >= 1)
            {
                vector norm_adjURes = L2norm(adjURes);
                scalar norm_adjpRes = L2norm(adjpRes);
                scalar norm_adjphiRes = L2norm(adjphiRes);
                scalar norm_adjnuTildaRes = L2norm(adjnuTildaRes);

                if (cnt == 1)
                {
                    initNormAdjURes = norm_adjURes;
                    initNormAdjpRes = norm_adjpRes;
                    initNormAdjphiRes = norm_adjphiRes;
                    initNormAdjnuTildaRes = norm_adjnuTildaRes;
                }

                // Normalize residual norms:
                norm_adjURes[0] /= initNormAdjURes[0];
                norm_adjURes[1] /= initNormAdjURes[1];
                norm_adjURes[2] /= initNormAdjURes[2];
                norm_adjpRes /= initNormAdjpRes;
                norm_adjphiRes /= initNormAdjphiRes;
                norm_adjnuTildaRes /= initNormAdjnuTildaRes;

                Info << "Normalized L2 norm of adjURes: " << norm_adjURes[0] << " " << norm_adjURes[1] << " " << norm_adjURes[2] << endl;
                Info << "Normalized L2 norm of adjpRes: " << norm_adjpRes << endl;
                Info << "Normalized L2 norm of adjphiRes: " << norm_adjphiRes << endl;
                Info << "Normalized L2 norm of adjnuTildaRes: " << norm_adjnuTildaRes << endl;

                // Check if fpRelTol is met:
                if (norm_adjURes[0] < fpRelTol && norm_adjURes[1] < fpRelTol && norm_adjURes[2] < fpRelTol && norm_adjpRes < fpRelTol && norm_adjphiRes < fpRelTol && norm_adjnuTildaRes < fpRelTol)
                {
                    Info << "Residual drop of " << fpRelTol << " has been achieved!" << endl;
                    break;
                }
            }

            cnt++;
        }

        // converged, assign the field Psi to psiVec
        this->vec2Fields("field2Vec", psi, UPsi, pPsi, phiPsi, nuTildaPsi);
    }
    else if (adjEqnSolMethod == "fixedPointC")
    {
        // not implemented yet
    }
    else
    {
        FatalErrorIn("adjEqnSolMethod not valid") << exit(FatalError);
    }

#endif
    return 0;
}

void DASimpleFoam::vec2Fields(
    const word mode,
    Vec cVec,
    volVectorField& UField,
    volScalarField& pField,
    surfaceScalarField& phiField,
    volScalarField& nuTildaField)
{
#ifdef CODI_AD_REVERSE
    PetscScalar* cVecArray;
    if (mode == "vec2Field")
    {
        VecGetArray(cVec, &cVecArray);

        // U
        forAll(meshPtr_->cells(), cellI)
        {
            for (label comp = 0; comp < 3; comp++)
            {
                label adjLocalIdx = daIndexPtr_->getLocalAdjointStateIndex("U", cellI, comp);
                UField[cellI][comp] = cVecArray[adjLocalIdx];
            }
        }
        // p
        forAll(meshPtr_->cells(), cellI)
        {
            label adjLocalIdx = daIndexPtr_->getLocalAdjointStateIndex("p", cellI);
            pField[cellI] = cVecArray[adjLocalIdx];
        }
        // phi
        forAll(meshPtr_->faces(), faceI)
        {
            label adjLocalIdx = daIndexPtr_->getLocalAdjointStateIndex("phi", faceI);

            if (faceI < daIndexPtr_->nLocalInternalFaces)
            {
                phiField[faceI] = cVecArray[adjLocalIdx];
            }
            else
            {
                label relIdx = faceI - daIndexPtr_->nLocalInternalFaces;
                label patchIdx = daIndexPtr_->bFacePatchI[relIdx];
                label faceIdx = daIndexPtr_->bFaceFaceI[relIdx];
                phiField.boundaryFieldRef()[patchIdx][faceIdx] = cVecArray[adjLocalIdx];
            }
        }
        // nuTilda
        forAll(meshPtr_->cells(), cellI)
        {
            label adjLocalIdx = daIndexPtr_->getLocalAdjointStateIndex("nuTilda", cellI);
            nuTildaField[cellI] = cVecArray[adjLocalIdx];
        }

        VecRestoreArray(cVec, &cVecArray);
    }
    else if (mode == "field2Vec")
    {
        VecGetArray(cVec, &cVecArray);

        // U
        forAll(meshPtr_->cells(), cellI)
        {
            for (label comp = 0; comp < 3; comp++)
            {
                label adjLocalIdx = daIndexPtr_->getLocalAdjointStateIndex("U", cellI, comp);
                cVecArray[adjLocalIdx] = UField[cellI][comp].value();
            }
        }
        // p
        forAll(meshPtr_->cells(), cellI)
        {
            label adjLocalIdx = daIndexPtr_->getLocalAdjointStateIndex("p", cellI);
            cVecArray[adjLocalIdx] = pField[cellI].value();
        }
        // phi
        forAll(meshPtr_->faces(), faceI)
        {
            label adjLocalIdx = daIndexPtr_->getLocalAdjointStateIndex("phi", faceI);

            if (faceI < daIndexPtr_->nLocalInternalFaces)
            {
                cVecArray[adjLocalIdx] = phiField[faceI].value();
            }
            else
            {
                label relIdx = faceI - daIndexPtr_->nLocalInternalFaces;
                label patchIdx = daIndexPtr_->bFacePatchI[relIdx];
                label faceIdx = daIndexPtr_->bFaceFaceI[relIdx];
                cVecArray[adjLocalIdx] = phiField.boundaryFieldRef()[patchIdx][faceIdx].value();
            }
        }
        // nuTilda
        forAll(meshPtr_->cells(), cellI)
        {
            label adjLocalIdx = daIndexPtr_->getLocalAdjointStateIndex("nuTilda", cellI);
            cVecArray[adjLocalIdx] = nuTildaField[cellI].value();
        }

        VecRestoreArray(cVec, &cVecArray);
    }
    else
    {
        FatalErrorIn("mode not valid") << exit(FatalError);
    }
#endif
}

void DASimpleFoam::invTranProdUEqn(
    const volVectorField& mySource,
    volVectorField& pseudoU)
{
    /*
    Description:
        Inverse transpose product, MU^(-T)
        Based on inverseProduct_UEqn from simpleFoamPrimal, but swaping upper() and lower()
        We won't ADR this function, so we can treat most of the arguments as const
    */

    const objectRegistry& db = meshPtr_->thisDb();
    const surfaceScalarField& phi = db.lookupObject<surfaceScalarField>("phi");
    volScalarField nuEff = daTurbulenceModelPtr_->nuEff();

    // Get the pseudoUEqn,
    // the most important thing here is to make sure the l.h.s. matches that of UEqn.
    fvVectorMatrix pseudoUEqn(
        fvm::div(phi, pseudoU)
        - fvm::laplacian(nuEff, pseudoU)
        - fvc::div(nuEff * dev2(T(fvc::grad(pseudoU)))));
    pseudoUEqn.relax();

    // Swap upper() and lower()
    List<scalar> temp = pseudoUEqn.upper();
    pseudoUEqn.upper() = pseudoUEqn.lower();
    pseudoUEqn.lower() = temp;

    // Overwrite the r.h.s.
    pseudoUEqn.source() = mySource;

    // Make sure that boundary contribution to source is zero,
    // Alternatively, we can deduct source by boundary contribution, so that it would cancel out during solve.
    forAll(pseudoU.boundaryField(), patchI)
    {
        const fvPatch& pp = pseudoU.boundaryField()[patchI].patch();
        forAll(pp, faceI)
        {
            label cellI = pp.faceCells()[faceI];
            pseudoUEqn.source()[cellI] -= pseudoUEqn.boundaryCoeffs()[patchI][faceI];
        }
    }

    // Before solve, force xEqn.psi() to be solved into all zero
    forAll(pseudoU.primitiveFieldRef(), cellI)
    {
        pseudoU.primitiveFieldRef()[cellI][0] = 0;
        pseudoU.primitiveFieldRef()[cellI][1] = 0;
        pseudoU.primitiveFieldRef()[cellI][2] = 0;
    }

    pseudoUEqn.solve();
}

void DASimpleFoam::invTranProdPEqn(
    const volScalarField& mySource,
    volScalarField& pseudoP)
{
    /*
    Description:
        Inverse transpose product, Mp^(-T)
        Based on inverseProduct_pEqn from simpleFoamPrimal, but swaping upper() and lower()
        We won't ADR this function, so we can treat most of the arguments as const
    */

    const objectRegistry& db = meshPtr_->thisDb();
    const volVectorField& U = db.lookupObject<volVectorField>("U");
    const surfaceScalarField& phi = db.lookupObject<surfaceScalarField>("phi");
    volScalarField nuEff = daTurbulenceModelPtr_->nuEff();

    // Construct UEqn first
    fvVectorMatrix UEqn(
        fvm::div(phi, U)
        - fvm::laplacian(nuEff, U)
        - fvc::div(nuEff * dev2(T(fvc::grad(U)))));
    // Without this, pRes would be way off.
    UEqn.relax();

    // create a scalar field with 1/A, reverse of A() of U
    volScalarField rAU(1.0 / UEqn.A());

    // Get the pseudoPEqn,
    // the most important thing here is to make sure the l.h.s. matches that of pEqn.
    fvScalarMatrix pseudoPEqn(fvm::laplacian(rAU, pseudoP));

    // Swap upper() and lower()
    List<scalar> temp = pseudoPEqn.upper();
    pseudoPEqn.upper() = pseudoPEqn.lower();
    pseudoPEqn.lower() = temp;

    // Overwrite the r.h.s.
    pseudoPEqn.source() = mySource;

    // pEqn.setReference(pRefCell, pRefValue);
    // Here, pRefCell is a label, and pRefValue is a scalar
    // In actual implementation, they need to passed into this function.
    pseudoPEqn.setReference(0, 0.0);

    // Make sure that boundary contribution to source is zero,
    // Alternatively, we can deduct source by boundary contribution, so that it would cancel out during solve.
    forAll(pseudoP.boundaryField(), patchI)
    {
        const fvPatch& pp = pseudoP.boundaryField()[patchI].patch();
        forAll(pp, faceI)
        {
            label cellI = pp.faceCells()[faceI];
            pseudoPEqn.source()[cellI] -= pseudoPEqn.boundaryCoeffs()[patchI][faceI];
        }
    }

    // Before solve, force xEqn.psi() to be solved into all zero
    forAll(pseudoP.primitiveFieldRef(), cellI)
    {
        pseudoP.primitiveFieldRef()[cellI] = 0;
    }

    pseudoPEqn.solve();
}

void DASimpleFoam::calcLduResiduals(
    volVectorField& URes,
    volScalarField& pRes,
    surfaceScalarField& phiRes)
{
    const objectRegistry& db = meshPtr_->thisDb();
    const volVectorField& U = db.lookupObject<volVectorField>("U");
    const volScalarField& p = db.lookupObject<volScalarField>("p");
    const surfaceScalarField& phi = db.lookupObject<surfaceScalarField>("phi");
    volScalarField nuEff = daTurbulenceModelPtr_->nuEff();

    fvVectorMatrix UEqn(
        fvm::div(phi, U)
        - fvm::laplacian(nuEff, U)
        - fvc::div(nuEff * dev2(T(fvc::grad(U))))); //This term is needed in res though...

    List<vector>& USource = UEqn.source();
    // Note we cannot use UEqn.D() here, because boundary contribution to diag have 3 components, and they can be different.
    // Thus we use UEqn.diag() here, and we correct both source and diag later.
    List<scalar>& UDiag = UEqn.diag();

    // Get fvc::grad(p), so that it can be added to r.h.s.
    volVectorField gradp(fvc::grad(p));

    // Initiate URes, with no boundary contribution
    for (label i = 0; i < U.size(); i++)
    {
        URes[i] = UDiag[i] * U[i] - USource[i] + U.mesh().V()[i] * gradp[i];
    }
    URes.primitiveFieldRef() -= UEqn.lduMatrix::H(U);

    // Add boundary contribution to source and diag
    // USource cannot be an alias, because it will be used downstream.
    forAll(U.boundaryField(), patchI)
    {
        const fvPatch& pp = U.boundaryField()[patchI].patch();
        forAll(pp, faceI)
        {
            // Both ways of getting cellI work
            // Below is the previous way of getting the address
            label cellI = pp.faceCells()[faceI];
            // Below is using lduAddr().patchAddr(patchi)
            //label cellI = UEqn.lduAddr().patchAddr(patchI)[faceI];
            for (label cmpt = 0; cmpt < 3; cmpt++)
            {
                URes[cellI][cmpt] += UEqn.internalCoeffs()[patchI][faceI][cmpt] * U[cellI][cmpt];
            }
            //Info << "UEqn.internalCoeffs()[" << patchI << "][" << faceI <<"]= " << UEqn.internalCoeffs()[patchI][faceI] <<endl;
            URes[cellI] -= UEqn.boundaryCoeffs()[patchI][faceI];
        }
    }

    // Below is not necessary, but it doesn't hurt
    URes.correctBoundaryConditions();

    UEqn.relax(); // Without this, pRes would be way off.

    volScalarField rAU(1.0 / UEqn.A()); // create a scalar field with 1/A, reverse of A() of U
    volVectorField HbyA("HbyA", U); // initialize a vector field with U and pass it to HbyA
    HbyA = rAU * UEqn.H(); // basically, HbyA = 1/A * H, H_by_A, need to verify source code though...
    surfaceScalarField phiHbyA("phiHbyA", fvc::flux(HbyA)); // get the flux of HbyA, phi_H_by_A

    fvScalarMatrix pEqn(
        fvm::laplacian(rAU, p) == fvc::div(phiHbyA));

    List<scalar>& pSource = pEqn.source();
    List<scalar>& pDiag = pEqn.diag();

    // Initiate pRes, with no boundary contribution
    for (label i = 0; i < p.size(); i++)
    {
        pRes[i] = pDiag[i] * p[i] - pSource[i];
    }
    pRes.primitiveFieldRef() -= pEqn.lduMatrix::H(p);

    // Boundary correction
    forAll(p.boundaryField(), patchI)
    {
        const fvPatch& pp = p.boundaryField()[patchI].patch();
        forAll(pp, faceI)
        {
            // Both ways of getting cellI work
            // Below is the previous way of getting the address
            label cellI = pp.faceCells()[faceI];
            // Below is using lduAddr().patchAddr(patchi)
            //label cellI = pEqn.lduAddr().patchAddr(patchI)[faceI];
            //myDiag[cellI] += TEqn.internalCoeffs()[patchI][faceI];
            pRes[cellI] += pEqn.internalCoeffs()[patchI][faceI] * p[cellI];
            pRes[cellI] -= pEqn.boundaryCoeffs()[patchI][faceI];
        }
    }

    // Below is not necessary, but it doesn't hurt
    pRes.correctBoundaryConditions();

    // Then do phiRes
    // Note: DAFoam also uses this formula for phiRes
    phiRes = phiHbyA - pEqn.flux() - phi;
}

void DASimpleFoam::calcAdjointResidual(
    volVectorField& URes,
    volScalarField& pRes,
    surfaceScalarField& phiRes,
    volScalarField& nuTildaRes,
    volVectorField& dfdU,
    volScalarField& dfdp,
    surfaceScalarField& dfdphi,
    volScalarField& dfdnuTilda,
    volVectorField& UPsi,
    volScalarField& pPsi,
    surfaceScalarField& phiPsi,
    volScalarField& nuTildaPsi,
    volVectorField& adjURes,
    volScalarField& adjpRes,
    surfaceScalarField& adjphiRes,
    volScalarField& adjnuTildaRes,
    label& cnt)
{
#ifdef CODI_AD_REVERSE
    volVectorField& U = const_cast<volVectorField&>(meshPtr_->thisDb().lookupObject<volVectorField>("U"));
    volScalarField& p = const_cast<volScalarField&>(meshPtr_->thisDb().lookupObject<volScalarField>("p"));
    volScalarField& nuTilda = const_cast<volScalarField&>(meshPtr_->thisDb().lookupObject<volScalarField>("nuTilda"));
    surfaceScalarField& phi = const_cast<surfaceScalarField&>(meshPtr_->thisDb().lookupObject<surfaceScalarField>("phi"));

    // Pass -dfdw to adjRes:
    adjURes = -dfdU;
    adjpRes = -dfdp;
    adjphiRes = -dfdphi;
    adjnuTildaRes = -dfdnuTilda;

    // get the reverse-mode AD tape
    codi::RealReverse::TapeType& tape = codi::RealReverse::getGlobalTape();

    if (cnt == 0)
    {
        tape.reset();
        tape.setActive();

        // register all (3+1) state variables as input
        // Start with U, note that U has 3 components
        forAll(U, cellI)
        {
            tape.registerInput(U[cellI][0]);
            tape.registerInput(U[cellI][1]);
            tape.registerInput(U[cellI][2]);
        }
        // Now register p as input:
        forAll(p, cellI)
        {
            tape.registerInput(p[cellI]);
        }
        // Then, register phi as input
        // Note that both the internal field of phi and the boundary fields need to be registered
        // Register internal of phi:
        forAll(phi.primitiveFieldRef(), faceI)
        {
            tape.registerInput(phi.primitiveFieldRef()[faceI]);
        }
        // Register boundary of phi:
        forAll(phi.boundaryFieldRef(), patchI)
        {
            forAll(phi.boundaryFieldRef()[patchI], faceI)
            {
                tape.registerInput(phi.boundaryFieldRef()[patchI][faceI]);
            }
        }
        // And then, register turbulence variable nuTilda as input:
        forAll(nuTilda, cellI)
        {
            tape.registerInput(nuTilda[cellI]);
        }

        // Correct boundaries to link the intermediate results
        U.correctBoundaryConditions();
        p.correctBoundaryConditions();
        nuTilda.correctBoundaryConditions();

        // Construct nuEff before calling lduCalcAllRes
        daTurbulenceModelPtr_->updateIntermediateVariables();

        // Call the residual functions
        this->calcLduResiduals(URes, pRes, phiRes);
        daTurbulenceModelPtr_->calcLduResidualTurb(nuTildaRes);

        // register output
        forAll(URes, cellI)
        {
            tape.registerOutput(URes[cellI][0]);
            tape.registerOutput(URes[cellI][1]);
            tape.registerOutput(URes[cellI][2]);
        }
        forAll(pRes, cellI)
        {
            tape.registerOutput(pRes[cellI]);
        }
        forAll(phiRes.primitiveFieldRef(), faceI)
        {
            tape.registerOutput(phiRes[faceI]);
        }
        // Seed boundary of phiRes:
        forAll(phiRes.boundaryFieldRef(), patchI)
        {
            forAll(phiRes.boundaryFieldRef()[patchI], faceI)
            {
                tape.registerOutput(phiRes.boundaryFieldRef()[patchI][faceI]);
            }
        }
        forAll(nuTildaRes, cellI)
        {
            tape.registerOutput(nuTildaRes[cellI]);
        }

        // stop recording
        tape.setPassive();
    }

    // set the AD seed to the output var
    // Start with URes, note that URes has 3 components
    forAll(URes, cellI)
    {
        URes[cellI][0].setGradient(UPsi[cellI][0].getValue());
        URes[cellI][1].setGradient(UPsi[cellI][1].getValue());
        URes[cellI][2].setGradient(UPsi[cellI][2].getValue());
    }
    // Now seed pRes:
    forAll(pRes, cellI)
    {
        pRes[cellI].setGradient(pPsi[cellI].getValue());
    }
    // Then, seed phiRes:
    // Seed internal of phiRes:
    forAll(phiRes.primitiveFieldRef(), faceI)
    {
        phiRes.primitiveFieldRef()[faceI].setGradient(phiPsi.primitiveFieldRef()[faceI].getValue());
    }
    // Seed boundary of phiRes:
    forAll(phiRes.boundaryFieldRef(), patchI)
    {
        forAll(phiRes.boundaryFieldRef()[patchI], faceI)
        {
            phiRes.boundaryFieldRef()[patchI][faceI].setGradient(phiPsi.boundaryFieldRef()[patchI][faceI].getValue());
        }
    }
    // And then, seed nuTildaRes:
    forAll(nuTildaRes, cellI)
    {
        nuTildaRes[cellI].setGradient(nuTildaPsi[cellI].getValue());
    }

    // evaluate the tape to compute the derivative of the seeded output wrt all the input
    tape.evaluate();
    forAll(U, cellI)
    {
        adjURes[cellI][0] += U[cellI][0].getGradient();
        adjURes[cellI][1] += U[cellI][1].getGradient();
        adjURes[cellI][2] += U[cellI][2].getGradient();
    }
    forAll(p, cellI)
    {
        adjpRes[cellI] += p[cellI].getGradient();
    }
    forAll(phi.primitiveFieldRef(), faceI)
    {
        adjphiRes.primitiveFieldRef()[faceI] += phi.primitiveFieldRef()[faceI].getGradient();
    }
    forAll(phi.boundaryFieldRef(), patchI)
    {
        forAll(phi.boundaryFieldRef()[patchI], faceI)
        {
            adjphiRes.boundaryFieldRef()[patchI][faceI] += phi.boundaryFieldRef()[patchI][faceI].getGradient();
        }
    }
    forAll(nuTilda, cellI)
    {
        adjnuTildaRes[cellI] += nuTilda[cellI].getGradient();
    }

    // Clear adjoints for future Jacobian calculations
    tape.clearAdjoints();
#endif
}
