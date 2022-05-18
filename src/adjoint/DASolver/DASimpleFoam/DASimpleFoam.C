/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v3

    This class is modified from OpenFOAM's source code
    applications/solvers/incompressible/simpleFoam

    OpenFOAM: The Open Source CFD Toolbox

    Copyright (C): 2011-2016 OpenFOAM Foundation

    OpenFOAM License:

        OpenFOAM is free software: you can redistribute it and/or modify it
        under the terms of the GNU General Public License as published by
        the Free Software Foundation, either version 3 of the License, or
        (at your option) any later version.
    
        OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
        ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
        FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
        for more details.
    
        You should have received a copy of the GNU General Public License
        along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

\*---------------------------------------------------------------------------*/

#include "DASimpleFoam.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(DASimpleFoam, 0);
addToRunTimeSelectionTable(DASolver, DASimpleFoam, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DASimpleFoam::DASimpleFoam(
    char* argsAll,
    PyObject* pyOptions)
    : DASolver(argsAll, pyOptions),
      simplePtr_(nullptr),
      pPtr_(nullptr),
      UPtr_(nullptr),
      phiPtr_(nullptr),
      alphaPorosityPtr_(nullptr),
      laminarTransportPtr_(nullptr),
      turbulencePtr_(nullptr),
      daTurbulenceModelPtr_(nullptr),
      daFvSourcePtr_(nullptr),
      fvSourcePtr_(nullptr),
      MRFPtr_(nullptr)
{
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

void DASimpleFoam::initSolver()
{
    /*
    Description:
        Initialize variables for DASolver
    */

    Info << "Initializing fields for DASimpleFoam" << endl;
    Time& runTime = runTimePtr_();
    fvMesh& mesh = meshPtr_();
#include "createSimpleControlPython.H"
#include "createFieldsSimple.H"
#include "createAdjointIncompressible.H"
    // initialize checkMesh
    daCheckMeshPtr_.reset(new DACheckMesh(daOptionPtr_(), runTime, mesh));

    daLinearEqnPtr_.reset(new DALinearEqn(mesh, daOptionPtr_()));

    this->setDAObjFuncList();

    // initialize fvSource and compute the source term
    const dictionary& allOptions = daOptionPtr_->getAllOptions();
    if (allOptions.subDict("fvSource").toc().size() != 0)
    {
        hasFvSource_ = 1;
        Info << "Initializing DASource" << endl;
        word sourceName = allOptions.subDict("fvSource").toc()[0];
        word fvSourceType = allOptions.subDict("fvSource").subDict(sourceName).getWord("type");
        daFvSourcePtr_.reset(DAFvSource::New(
            fvSourceType, mesh, daOptionPtr_(), daModelPtr_(), daIndexPtr_()));
    }
}

label DASimpleFoam::solvePrimal(
    const Vec xvVec,
    Vec wVec)
{
    /*
    Description:
        Call the primal solver to get converged state variables

    Input:
        xvVec: a vector that contains all volume mesh coordinates

    Output:
        wVec: state variable vector
    */

#include "createRefsSimple.H"
#include "createFvOptions.H"

    // change the run status
    daOptionPtr_->setOption<word>("runStatus", "solvePrimal");

    // call correctNut, this is equivalent to turbulence->validate();
    daTurbulenceModelPtr_->updateIntermediateVariables();

    Info << "\nStarting time loop\n"
         << endl;

    // deform the mesh based on the xvVec
    this->pointVec2OFMesh(xvVec);

    // check mesh quality
    label meshOK = this->checkMesh();

    if (!meshOK)
    {
        this->writeFailedMesh();
        return 1;
    }

    // if the forwardModeAD is active, we need to set the seed here
#include "setForwardADSeeds.H"

    word divUScheme = "div(phi,U)";
    if (daOptionPtr_->getSubDictOption<label>("runLowOrderPrimal4PC", "active"))
    {
        if (daOptionPtr_->getSubDictOption<label>("runLowOrderPrimal4PC", "isPC"))
        {
            Info << "Using low order div scheme for primal solution .... " << endl;
            divUScheme = "div(pc)";
        }
    }

    primalMinRes_ = 1e10;
    label printInterval = daOptionPtr_->getOption<label>("printInterval");
    label printToScreen = 0;
    while (this->loop(runTime)) // using simple.loop() will have seg fault in parallel
    {

        printToScreen = this->isPrintTime(runTime, printInterval);

        if (printToScreen)
        {
            Info << "Time = " << runTime.timeName() << nl << endl;
        }

        p.storePrevIter();

        // --- Pressure-velocity SIMPLE corrector
        {
#include "UEqnSimple.H"
#include "pEqnSimple.H"
        }

        laminarTransport.correct();
        daTurbulenceModelPtr_->correct();

        if (printToScreen)
        {
            daTurbulenceModelPtr_->printYPlus();

            this->printAllObjFuncs();

            Info << "ExecutionTime = " << runTime.elapsedCpuTime() << " s"
                 << "  ClockTime = " << runTime.elapsedClockTime() << " s"
                 << nl << endl;
        }

        runTime.write();
    }

    this->writeAssociatedFields();

    this->calcPrimalResidualStatistics("print");

    // primal converged, assign the OpenFoam fields to the state vec wVec
    this->ofField2StateVec(wVec);

    // write the mesh to files
    mesh.write();

    Info << "End\n"
         << endl;

    return this->checkResidualTol();
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

        for (label n = 0; n < fpMaxIters; n++)
        {
            Info << "Time Step: " << n << "        Execution Time: " << meshPtr_->time().elapsedCpuTime() << " s" << endl;

            // ************ U **************
            // now calculate the residual
            this->calcdRdWTPsiAD(0, psi, adjRes);
            VecAYPX(adjRes, -1.0, dFdW);

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
                    psiArray[adjLocalIdx] += dPsiU[cellI][comp].value();
                }
            }
            VecRestoreArray(psi, &psiArray);

            // ************ p **************
            // now calculate the residual
            this->calcdRdWTPsiAD(0, psi, adjRes);
            VecAYPX(adjRes, -1.0, dFdW);

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
                psiArray[adjLocalIdx] += dPsiP[cellI].value();
            }
            VecRestoreArray(psi, &psiArray);

            // ************ phi **************
            // now calculate the residual
            this->calcdRdWTPsiAD(0, psi, adjRes);
            VecAYPX(adjRes, -1.0, dFdW);

            // assign adjRes to adjPhiRes
            VecGetArrayRead(adjRes, &adjResArray);
            VecGetArray(psi, &psiArray);
            forAll(meshPtr_->faces(), faceI)
            {
                label adjLocalIdx = daIndexPtr_->getLocalAdjointStateIndex("phi", faceI);
                psiArray[adjLocalIdx] += adjResArray[adjLocalIdx];
            }
            VecRestoreArray(psi, &psiArray);
            VecRestoreArrayRead(adjRes, &adjResArray);

            // ************ nuTilda **************
            // now calculate the residual
            this->calcdRdWTPsiAD(0, psi, adjRes);
            VecAYPX(adjRes, -1.0, dFdW);

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
                psiArray[adjLocalIdx] += dPsiNuTilda[cellI].value();
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

} // End namespace Foam

// ************************************************************************* //
