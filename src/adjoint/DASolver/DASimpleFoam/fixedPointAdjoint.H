/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v3

    Fixed-point adjoint members

\*---------------------------------------------------------------------------*/

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

        const objectRegistry& db = meshPtr_->thisDb();
        const volVectorField& U = db.lookupObject<volVectorField>("U");
        const volScalarField& p = db.lookupObject<volScalarField>("p");
        const surfaceScalarField& phi = db.lookupObject<surfaceScalarField>("phi");
        const volScalarField& nuTilda = db.lookupObject<volScalarField>("nuTilda");

        // adjoint residuals for all vars
        Vec adjRes;
        VecDuplicate(dFdW, &adjRes);
        VecZeroEntries(adjRes);
        // adjoint residual array
        const PetscScalar* adjResArray;
        // adjoint residual for U
        List<vector> adjURes(meshPtr_->nCells(), vector::zero);
        // adjoint residual for p
        List<scalar> adjPRes(meshPtr_->nCells(), 0.0);
        // adjoint residual for nuTilda
        List<scalar> adjNuTildaRes(meshPtr_->nCells(), 0.0);

        // adjoint var psi array
        PetscScalar* psiArray;

        // delta psi for U
        volVectorField dPsiU("dPsiU", U);
        volScalarField dPsiP("dPsiP", p);
        volScalarField dPsiNuTilda("dPsiNuTilda", nuTilda);

        // calculate the initial residual and record the tape: R = -dR/dWT*psi + dF/dW
        this->calcdRdWTPsiAD(1, psi, adjRes);
        VecAYPX(adjRes, -1.0, dFdW);

        PetscScalar adjResNorm;
        for (label n = 0; n < fpMaxIters; n++)
        {
            Info << "Time Step: " << n << "        Execution Time: " << meshPtr_->time().elapsedCpuTime() << " s" << endl;

            // ************ U **************
            // now calculate the residual
            this->calcdRdWTPsiAD(0, psi, adjRes);
            VecAYPX(adjRes, -1.0, dFdW);

            VecNorm(adjRes, NORM_2, &adjResNorm);
            Info << "adjResNorm: " << adjResNorm << endl;

            // assign adjRes to adjURes
            VecGetArrayRead(adjRes, &adjResArray);
            forAll(meshPtr_->cells(), cellI)
            {
                for (label comp = 0; comp < 3; comp++)
                {
                    label adjLocalIdx = daIndexPtr_->getLocalAdjointStateIndex("U", cellI, comp);
                    adjURes[cellI][comp] = adjResArray[adjLocalIdx];
                }
            }
            VecRestoreArrayRead(adjRes, &adjResArray);

            // calculate the dPsiU
            invTranProdUEqn(adjURes, dPsiU);

            // now add dPsiU to psi
            VecGetArray(psi, &psiArray);
            forAll(meshPtr_->cells(), cellI)
            {
                for (label comp = 0; comp < 3; comp++)
                {
                    label adjLocalIdx = daIndexPtr_->getLocalAdjointStateIndex("U", cellI, comp);
                    psiArray[adjLocalIdx] += relaxU.value() * dPsiU[cellI][comp].value();
                }
            }
            VecRestoreArray(psi, &psiArray);

            // ************ p **************
            // now calculate the residual
            this->calcdRdWTPsiAD(0, psi, adjRes);
            VecAYPX(adjRes, -1.0, dFdW);

            VecNorm(adjRes, NORM_2, &adjResNorm);
            Info << "adjResNorm: " << adjResNorm << endl;

            // assign adjRes to adjPRes
            VecGetArrayRead(adjRes, &adjResArray);
            forAll(meshPtr_->cells(), cellI)
            {
                label adjLocalIdx = daIndexPtr_->getLocalAdjointStateIndex("p", cellI);
                adjPRes[cellI] = adjResArray[adjLocalIdx];
            }
            VecRestoreArrayRead(adjRes, &adjResArray);

            // calculate the dPsiP
            invTranProdPEqn(adjPRes, dPsiP);

            // now add dPsiP to psi
            VecGetArray(psi, &psiArray);
            forAll(meshPtr_->cells(), cellI)
            {
                label adjLocalIdx = daIndexPtr_->getLocalAdjointStateIndex("p", cellI);
                psiArray[adjLocalIdx] += relaxP.value() * dPsiP[cellI].value();
            }
            VecRestoreArray(psi, &psiArray);

            // ************ phi **************
            // now calculate the residual
            this->calcdRdWTPsiAD(0, psi, adjRes);
            VecAYPX(adjRes, -1.0, dFdW);

            VecNorm(adjRes, NORM_2, &adjResNorm);
            Info << "adjResNorm: " << adjResNorm << endl;

            // assign adjRes to adjPhiRes
            VecGetArrayRead(adjRes, &adjResArray);
            VecGetArray(psi, &psiArray);
            forAll(meshPtr_->faces(), faceI)
            {
                label adjLocalIdx = daIndexPtr_->getLocalAdjointStateIndex("phi", faceI);
                psiArray[adjLocalIdx] -= relaxPhi.value() * adjResArray[adjLocalIdx];
            }
            VecRestoreArray(psi, &psiArray);
            VecRestoreArrayRead(adjRes, &adjResArray);

            // ************ nuTilda **************
            // now calculate the residual
            this->calcdRdWTPsiAD(0, psi, adjRes);
            VecAYPX(adjRes, -1.0, dFdW);

            VecNorm(adjRes, NORM_2, &adjResNorm);
            Info << "adjResNorm: " << adjResNorm << endl;

            // assign adjRes to adjNuTildaRes
            VecGetArrayRead(adjRes, &adjResArray);
            forAll(meshPtr_->cells(), cellI)
            {
                label adjLocalIdx = daIndexPtr_->getLocalAdjointStateIndex("nuTilda", cellI);
                adjNuTildaRes[cellI] = adjResArray[adjLocalIdx];
            }
            VecRestoreArrayRead(adjRes, &adjResArray);

            // calculate the dPsiNuTilda
            daTurbulenceModelPtr_->invTranProdNuTildaEqn(adjNuTildaRes, dPsiNuTilda);

            // now add dPsiNuTilda to psi
            VecGetArray(psi, &psiArray);
            forAll(meshPtr_->cells(), cellI)
            {
                label adjLocalIdx = daIndexPtr_->getLocalAdjointStateIndex("p", cellI);
                psiArray[adjLocalIdx] += relaxNuTilda.value() * dPsiNuTilda[cellI].value();
            }
            VecRestoreArray(psi, &psiArray);
        }
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

void DASimpleFoam::invTranProdUEqn(
    const List<vector>& mySource,
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
    const List<scalar>& mySource,
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
    List<vector>& dfdU,
    List<scalar>& dfdp,
    surfaceScalarField& dfdphi,
    List<scalar>& dfdnuTilda,
    List<vector>& UPsi,
    List<scalar>& pPsi,
    surfaceScalarField& phiPsi,
    List<scalar>& nuTildaPsi,
    List<vector>& adjURes,
    List<scalar>& adjpRes,
    surfaceScalarField& adjphiRes,
    List<scalar>& adjnuTildaRes,
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
            tape.registerOutput(nuTilda[cellI]);
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
