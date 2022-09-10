/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v3

    Fixed-point adjoint members

\*---------------------------------------------------------------------------*/

// L2 norm of a scalar list (or the primitive field of a volScalarField)
// Scale off cell volumes
scalar DASimpleFoam::L2norm(const volScalarField& v)
{
    scalar L2normV = 0.0;

    forAll(v, cellI)
    {
        L2normV += sqr(v[cellI] / meshPtr_->V()[cellI]);
    }
    L2normV = sqrt(L2normV);

    return L2normV;
}

// Scale off cell volumes
vector DASimpleFoam::L2norm(const volVectorField& U)
{
    vector L2normU = vector::zero;

    forAll(U, cellI)
    {
        L2normU[0] += sqr(U[cellI][0] / meshPtr_->V()[cellI]);
        L2normU[1] += sqr(U[cellI][1] / meshPtr_->V()[cellI]);
        L2normU[2] += sqr(U[cellI][2] / meshPtr_->V()[cellI]);
    }
    L2normU[0] = sqrt(L2normU[0]);
    L2normU[1] = sqrt(L2normU[1]);
    L2normU[2] = sqrt(L2normU[2]);

    return L2normU;
}

// L2 norm of a surfaceScalarField
scalar DASimpleFoam::L2norm(const surfaceScalarField& Phi)
{
    scalar L2normPhi = 0.0;

    forAll(Phi.primitiveField(), cellI)
    {
        L2normPhi += sqr(Phi.primitiveField()[cellI]);
    }
    forAll(Phi.boundaryField(), patchI)
    {
        forAll(Phi.boundaryField()[patchI], faceI)
        {
            L2normPhi += sqr(Phi.boundaryField()[patchI][faceI]);
        }
    }
    L2normPhi = sqrt(L2normPhi);

    return L2normPhi;
}

// This function is for swapping the upper and lower of a xxEqn
void DASimpleFoam::swap(List<scalar>& a, List<scalar>& b)
{
    List<scalar> temp = a;
    a = b;
    b = temp;
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

        volVectorField pseudoU("pseudoU", U);
        volScalarField pseudoP("pseudoP", p);
        volScalarField pseudoNuTilda("pseudoNuTilda", nuTilda);

        volVectorField adjURes("adjURes", 0.0 * U);
        volScalarField adjPRes("adjPRes", 0.0 * p);
        surfaceScalarField adjPhiRes("adjPhiRes", 0.0 * phi);
        volScalarField adjNuTildaRes("adjNuTildaRes", 0.0 * nuTilda);

        // Initiate the initial L2norm for adjURes, adjPRes, adjPhiRes, adjNuTildaRes
        vector initNormAdjURes = vector::zero;
        scalar initNormAdjPRes = 0.0;
        scalar initNormAdjPhiRes = 0.0;
        scalar initNormAdjNuTildaRes = 0.0;

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
            this->calcAdjointResidual(URes, pRes, phiRes, nuTildaRes, dFdU, dFdP, dFdPhi, dFdNuTilda, UPsi, pPsi, phiPsi, nuTildaPsi, adjURes, adjPRes, adjPhiRes, adjNuTildaRes, cnt);
            invTranProdUEqn(adjURes, pseudoU);
            forAll(pseudoU, cellI)
            {
                // the definition of adjURes has opposite sign than the formulation on Overleaf, so we do -= here
                UPsi[cellI] -= relaxU * pseudoU[cellI];
            }

            //  ------- p ----
            this->calcAdjointResidual(URes, pRes, phiRes, nuTildaRes, dFdU, dFdP, dFdPhi, dFdNuTilda, UPsi, pPsi, phiPsi, nuTildaPsi, adjURes, adjPRes, adjPhiRes, adjNuTildaRes, cnt);
            invTranProdPEqn(adjPRes, pseudoP);
            forAll(pseudoP, cellI)
            {
                pPsi[cellI] -= relaxP * pseudoP[cellI];
            }

            // -------- phi  --------
            this->calcAdjointResidual(URes, pRes, phiRes, nuTildaRes, dFdU, dFdP, dFdPhi, dFdNuTilda, UPsi, pPsi, phiPsi, nuTildaPsi, adjURes, adjPRes, adjPhiRes, adjNuTildaRes, cnt);

            forAll(adjPhiRes, faceI)
            {
                phiPsi[faceI] += relaxPhi * adjPhiRes[faceI];
            }
            forAll(adjPhiRes.boundaryField(), patchI)
            {
                forAll(adjPhiRes.boundaryField()[patchI], faceI)
                {
                    phiPsi.boundaryFieldRef()[patchI][faceI] += relaxPhi * adjPhiRes.boundaryField()[patchI][faceI];
                }
            }

            // -------- nuTilda  --------
            this->calcAdjointResidual(URes, pRes, phiRes, nuTildaRes, dFdU, dFdP, dFdPhi, dFdNuTilda, UPsi, pPsi, phiPsi, nuTildaPsi, adjURes, adjPRes, adjPhiRes, adjNuTildaRes, cnt);
            // Solve, using the inverse transpose product function
            daTurbulenceModelPtr_->invTranProdNuTildaEqn(adjNuTildaRes, pseudoNuTilda);

            forAll(pseudoNuTilda, cellI)
            {
                nuTildaPsi[cellI] -= relaxNuTilda * pseudoNuTilda[cellI];
            }

            if (cnt >= 1)
            {
                vector normAdjURes = L2norm(adjURes);
                scalar normAdjPRes = L2norm(adjPRes);
                scalar normAdjPhiRes = L2norm(adjPhiRes);
                scalar normAdjNuTildaRes = L2norm(adjNuTildaRes);

                if (cnt == 1)
                {
                    initNormAdjURes = normAdjURes;
                    initNormAdjPRes = normAdjPRes;
                    initNormAdjPhiRes = normAdjPhiRes;
                    initNormAdjNuTildaRes = normAdjNuTildaRes;
                }

                // Normalize residual norms:
                normAdjURes[0] /= initNormAdjURes[0];
                normAdjURes[1] /= initNormAdjURes[1];
                normAdjURes[2] /= initNormAdjURes[2];
                normAdjPRes /= initNormAdjPRes;
                normAdjPhiRes /= initNormAdjPhiRes;
                normAdjNuTildaRes /= initNormAdjNuTildaRes;

                Info << "Normalized L2 norm of adjURes: " << normAdjURes[0] << " " << normAdjURes[1] << " " << normAdjURes[2] << endl;
                Info << "Normalized L2 norm of adjPRes: " << normAdjPRes << endl;
                Info << "Normalized L2 norm of adjPhiRes: " << normAdjPhiRes << endl;
                Info << "Normalized L2 norm of adjNuTildaRes: " << normAdjNuTildaRes << endl;

                // Check if fpRelTol is met:
                if (normAdjURes[0] < fpRelTol && normAdjURes[1] < fpRelTol && normAdjURes[2] < fpRelTol && normAdjPRes < fpRelTol && normAdjPhiRes < fpRelTol && normAdjNuTildaRes < fpRelTol)
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

    // Consistent fixed-point adjoint
    else if (adjEqnSolMethod == "fixedPointC")
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
        // better do it another way!
        volScalarField nuEff = daTurbulenceModelPtr_->nuEff();

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

        // Duplicate U, p, nuTilda to get pseudo variables
        volVectorField pseudoU("pseudoU", U);
        volScalarField pseudoP("pseudoP", p);
        volScalarField pseudoNuTilda("pseudoNuTilda", nuTilda);

        // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
        // Get the pseudoUEqn,
        // temporarily in main code, pack into a class variable later
        fvVectorMatrix pseudoUEqn(
            fvm::div(phi, pseudoU)
            - fvm::laplacian(nuEff, pseudoU));
        pseudoUEqn.relax();

        // Swap upper() and lower()
        swap(pseudoUEqn.upper(), pseudoUEqn.lower());
        // Get UDiag as D
        List<scalar> UDiag = pseudoUEqn.D();
        // Get rAU
        volScalarField rAU(1.0 / pseudoUEqn.A());

        // *********************************** //
        // Get the pseudoPEqn,
        // temporarily in main code, pack into a class variable later
        fvScalarMatrix pseudoPEqn(
            fvm::laplacian(rAU, pseudoP));
        // Swap upper() and lower()
        swap(pseudoPEqn.upper(), pseudoPEqn.lower());

        // *********************************** //
        // Get the pseudoNuTildaEqn,
        // Add to DASpalartAllmarasFv3.C and .H
        /*
        const volScalarField chi(daTurbulenceModelPtr_->chi());
        const volScalarField fv1(daTurbulenceModelPtr_->fv1(chi));

        // Get myStilda
        // ??? How to access kappa and y?
        const volScalarField Stilda(
        daTurbulenceModelPtr_->fv3(chi, fv1) * ::sqrt(2.0) * mag(skew(fvc::grad(U)))
        + daTurbulenceModelPtr_->fv2(chi, fv1) * nuTilda / sqr(kappa_ * y_));

        // Get the pseudoNuTildaEqn,
        fvScalarMatrix pseudoNuTildaEqn(
            fvm::ddt(pseudoNuTilda)
                + fvm::div(phi, pseudoNuTilda)
                - fvm::laplacian(DnuTildaEff(pseudoNuTilda, nu, sigmaNut), pseudoNuTilda)
            == -fvm::Sp(Cw1 * fw(myStilda, pseudoNuTilda, y, kappa, Cw2, Cw3) * pseudoNuTilda / sqr(y), pseudoNuTilda));
        pseudoNuTildaEqn.relax();

        // Swap upper() and lower()
        swap(pseudoNuTildaEqn.upper(), pseudoNuTildaEqn.lower());
        */
        // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

        volVectorField adjURes("adjURes", 0.0 * U);
        volScalarField adjPRes("adjPRes", 0.0 * p);
        surfaceScalarField adjPhiRes("adjPhiRes", 0.0 * phi);
        volScalarField adjNuTildaRes("adjNuTildaRes", 0.0 * nuTilda);

        // Initiate the initial L2norm for adjURes, adjPRes, adjPhiRes, adjNuTildaRes
        vector initNormAdjURes = vector::zero;
        scalar initNormAdjPRes = 0.0;
        scalar initNormAdjPhiRes = 0.0;
        scalar initNormAdjNuTildaRes = 0.0;

        this->calcLduResiduals(URes, pRes, phiRes);
        daTurbulenceModelPtr_->calcLduResidualTurb(nuTildaRes);
        Info << "Residual for simpleFOAM after convergence: " << endl;
        Info << "L2 norm of URes: " << L2norm(URes) << endl;
        Info << "L2 norm of pRes: " << L2norm(pRes) << endl;
        Info << "L2 norm of phiRes: " << L2norm(phiRes) << endl;
        Info << "L2 norm of nuTildaRes: " << L2norm(nuTildaRes) << endl;

        // Initialize the duplicated variables for AD
        volScalarField p1("p1", p);
        volScalarField p2("p2", p);
        this->meshPtr_->setFluxRequired(p2.name());
        surfaceScalarField phi1("phi1", phi);
        volVectorField U1("U1", U);

        // Initialize the intermidiate variables for AD
        volVectorField gradP("gradP", fvc::grad(p));
        surfaceScalarField pEqnFlux("pEqnFlux", 0.0 * phi);
        volScalarField divPhi("divPhi", fvc::div(phi));
        surfaceScalarField fluxU("fluxU", 0.0 * phi);

        label cnt = 0;
        while (cnt < fpMaxIters)
        {
            Info << "Step = " << cnt << "  Execution Time: " << meshPtr_->time().elapsedCpuTime() << " s" << endl;

            // Placeholders
            /*
            // ------ U -----
            this->calcAdjointResidual(URes, pRes, phiRes, nuTildaRes, dFdU, dFdP, dFdPhi, dFdNuTilda, UPsi, pPsi, phiPsi, nuTildaPsi, adjURes, adjPRes, adjPhiRes, adjNuTildaRes, cnt);
            invTranProdUEqn(adjURes, pseudoU);
            forAll(pseudoU, cellI)
            {
                // the definition of adjURes has opposite sign than the formulation on Overleaf, so we do -= here
                UPsi[cellI] -= relaxU * pseudoU[cellI];
            }

            //  ------- p ----
            this->calcAdjointResidual(URes, pRes, phiRes, nuTildaRes, dFdU, dFdP, dFdPhi, dFdNuTilda, UPsi, pPsi, phiPsi, nuTildaPsi, adjURes, adjPRes, adjPhiRes, adjNuTildaRes, cnt);
            invTranProdPEqn(adjPRes, pseudoP);
            forAll(pseudoP, cellI)
            {
                pPsi[cellI] -= relaxP * pseudoP[cellI];
            }

            // -------- phi  --------
            this->calcAdjointResidual(URes, pRes, phiRes, nuTildaRes, dFdU, dFdP, dFdPhi, dFdNuTilda, UPsi, pPsi, phiPsi, nuTildaPsi, adjURes, adjPRes, adjPhiRes, adjNuTildaRes, cnt);

            forAll(adjPhiRes, faceI)
            {
                phiPsi[faceI] += relaxPhi * adjPhiRes[faceI];
            }
            forAll(adjPhiRes.boundaryField(), patchI)
            {
                forAll(adjPhiRes.boundaryField()[patchI], faceI)
                {
                    phiPsi.boundaryFieldRef()[patchI][faceI] += relaxPhi * adjPhiRes.boundaryField()[patchI][faceI];
                }
            }

            // -------- nuTilda  --------
            this->calcAdjointResidual(URes, pRes, phiRes, nuTildaRes, dFdU, dFdP, dFdPhi, dFdNuTilda, UPsi, pPsi, phiPsi, nuTildaPsi, adjURes, adjPRes, adjPhiRes, adjNuTildaRes, cnt);
            // Solve, using the inverse transpose product function
            daTurbulenceModelPtr_->invTranProdNuTildaEqn(adjNuTildaRes, pseudoNuTilda);

            forAll(pseudoNuTilda, cellI)
            {
                nuTildaPsi[cellI] -= relaxNuTilda * pseudoNuTilda[cellI];
            }
            */

            if (cnt >= 1)
            {
                vector normAdjURes = L2norm(adjURes);
                scalar normAdjPRes = L2norm(adjPRes);
                scalar normAdjPhiRes = L2norm(adjPhiRes);
                scalar normAdjNuTildaRes = L2norm(adjNuTildaRes);

                if (cnt == 1)
                {
                    initNormAdjURes = normAdjURes;
                    initNormAdjPRes = normAdjPRes;
                    initNormAdjPhiRes = normAdjPhiRes;
                    initNormAdjNuTildaRes = normAdjNuTildaRes;
                }

                // Normalize residual norms:
                normAdjURes[0] /= initNormAdjURes[0];
                normAdjURes[1] /= initNormAdjURes[1];
                normAdjURes[2] /= initNormAdjURes[2];
                normAdjPRes /= initNormAdjPRes;
                normAdjPhiRes /= initNormAdjPhiRes;
                normAdjNuTildaRes /= initNormAdjNuTildaRes;

                Info << "Normalized L2 norm of adjURes: " << normAdjURes[0] << " " << normAdjURes[1] << " " << normAdjURes[2] << endl;
                Info << "Normalized L2 norm of adjPRes: " << normAdjPRes << endl;
                Info << "Normalized L2 norm of adjPhiRes: " << normAdjPhiRes << endl;
                Info << "Normalized L2 norm of adjNuTildaRes: " << normAdjNuTildaRes << endl;

                // Check if fpRelTol is met:
                if (normAdjURes[0] < fpRelTol && normAdjURes[1] < fpRelTol && normAdjURes[2] < fpRelTol && normAdjPRes < fpRelTol && normAdjPhiRes < fpRelTol && normAdjNuTildaRes < fpRelTol)
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
    volVectorField gradP(fvc::grad(p));

    // Initiate URes, with no boundary contribution
    for (label i = 0; i < U.size(); i++)
    {
        URes[i] = UDiag[i] * U[i] - USource[i] + U.mesh().V()[i] * gradP[i];
    }
    URes.primitiveFieldRef() -= UEqn.lduMatrix::H(U);

    // Add boundary contribution to source and diag
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
    volVectorField& dFdU,
    volScalarField& dFdP,
    surfaceScalarField& dFdPhi,
    volScalarField& dFdNuTilda,
    volVectorField& UPsi,
    volScalarField& pPsi,
    surfaceScalarField& phiPsi,
    volScalarField& nuTildaPsi,
    volVectorField& adjURes,
    volScalarField& adjPRes,
    surfaceScalarField& adjPhiRes,
    volScalarField& adjNuTildaRes,
    label& cnt)
{
#ifdef CODI_AD_REVERSE
    volVectorField& U = const_cast<volVectorField&>(meshPtr_->thisDb().lookupObject<volVectorField>("U"));
    volScalarField& p = const_cast<volScalarField&>(meshPtr_->thisDb().lookupObject<volScalarField>("p"));
    volScalarField& nuTilda = const_cast<volScalarField&>(meshPtr_->thisDb().lookupObject<volScalarField>("nuTilda"));
    surfaceScalarField& phi = const_cast<surfaceScalarField&>(meshPtr_->thisDb().lookupObject<surfaceScalarField>("phi"));

    // Pass -dfdw to adjRes:
    adjURes = -dFdU;
    adjPRes = -dFdP;
    adjPhiRes = -dFdPhi;
    adjNuTildaRes = -dFdNuTilda;

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
        adjPRes[cellI] += p[cellI].getGradient();
    }
    forAll(phi.primitiveFieldRef(), faceI)
    {
        adjPhiRes.primitiveFieldRef()[faceI] += phi.primitiveFieldRef()[faceI].getGradient();
    }
    forAll(phi.boundaryFieldRef(), patchI)
    {
        forAll(phi.boundaryFieldRef()[patchI], faceI)
        {
            adjPhiRes.boundaryFieldRef()[patchI][faceI] += phi.boundaryFieldRef()[patchI][faceI].getGradient();
        }
    }
    forAll(nuTilda, cellI)
    {
        adjNuTildaRes[cellI] += nuTilda[cellI].getGradient();
    }

    // Clear adjoints for future Jacobian calculations
    tape.clearAdjoints();
#endif
}
