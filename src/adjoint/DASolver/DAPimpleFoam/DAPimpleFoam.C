/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

    This class is modified from OpenFOAM's source code
    applications/solvers/incompressible/pimpleFoam

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

#include "DAPimpleFoam.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(DAPimpleFoam, 0);
addToRunTimeSelectionTable(DASolver, DAPimpleFoam, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DAPimpleFoam::DAPimpleFoam(
    char* argsAll,
    PyObject* pyOptions)
    : DASolver(argsAll, pyOptions),
      pimplePtr_(nullptr),
      pPtr_(nullptr),
      UPtr_(nullptr),
      phiPtr_(nullptr),
      laminarTransportPtr_(nullptr),
      turbulencePtr_(nullptr),
      daTurbulenceModelPtr_(nullptr),
      daFvSourcePtr_(nullptr),
      fvSourcePtr_(nullptr),
      PrPtr_(nullptr),
      PrtPtr_(nullptr),
      TPtr_(nullptr),
      alphatPtr_(nullptr)
{
    // check whether the temperature field exists in the 0 folder
    hasTField_ = DAUtility::isFieldReadable(meshPtr_(), "T", "volScalarField");
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

void DAPimpleFoam::initSolver()
{
    /*
    Description:
        Initialize variables for DASolver
    */

    Info << "Initializing fields for DAPimpleFoam" << endl;
    Time& runTime = runTimePtr_();
    fvMesh& mesh = meshPtr_();
#include "createPimpleControlPython.H"
#include "createFieldsPimple.H"

    // read the RAS model from constant/turbulenceProperties
    const word turbModelName(
        IOdictionary(
            IOobject(
                "turbulenceProperties",
                mesh.time().constant(),
                mesh,
                IOobject::MUST_READ,
                IOobject::NO_WRITE,
                false))
            .subDict("RAS")
            .lookup("RASModel"));
    daTurbulenceModelPtr_.reset(DATurbulenceModel::New(turbModelName, mesh, daOptionPtr_()));

#include "createAdjoint.H"

    // initialize fvSource and the source term
    const dictionary& allOptions = daOptionPtr_->getAllOptions();
    if (allOptions.subDict("fvSource").toc().size() != 0)
    {
        hasFvSource_ = 1;
        Info << "Computing fvSource" << endl;
        word sourceName = allOptions.subDict("fvSource").toc()[0];
        word fvSourceType = allOptions.subDict("fvSource").subDict(sourceName).getWord("type");
        daFvSourcePtr_.reset(DAFvSource::New(
            fvSourceType, mesh, daOptionPtr_(), daModelPtr_(), daIndexPtr_()));
        daFvSourcePtr_->calcFvSource(fvSource);
    }

    // reduceIO does not write mesh, but if there is a shape variable, set writeMesh to 1
    dictionary dvSubDict = daOptionPtr_->getAllOptions().subDict("inputInfo");
    forAll(dvSubDict.toc(), idxI)
    {
        word dvName = dvSubDict.toc()[idxI];
        if (dvSubDict.subDict(dvName).getWord("type") == "volCoord")
        {
            reduceIOWriteMesh_ = 1;
            break;
        }
    }
}

label DAPimpleFoam::solvePrimal()
{
    /*
    Description:
        Call the primal solver to get converged state variables
    */

#include "createRefsPimple.H"

    // call correctNut, this is equivalent to turbulence->validate();
    daTurbulenceModelPtr_->updateIntermediateVariables();

    Info << "\nStarting time loop\n"
         << endl;

    label pimplePrintToScreen = 0;

    // we need to reduce the number of files written to the disk to minimize the file IO load
    label reduceIO = daOptionPtr_->getAllOptions().subDict("unsteadyAdjoint").getLabel("reduceIO");
    wordList additionalOutput;
    if (reduceIO)
    {
        daOptionPtr_->getAllOptions().subDict("unsteadyAdjoint").readEntry<wordList>("additionalOutput", additionalOutput);
    }

    scalar endTime = runTime.endTime().value();
    scalar deltaT = runTime.deltaT().value();
    label nInstances = round(endTime / deltaT);

    // main loop
    label regModelFail = 0;
    label fail = 0;
    for (label iter = 1; iter <= nInstances; iter++)
    {
        ++runTime;

        // if we have unsteadyField in inputInfo, assign GlobalVar::inputFieldUnsteady to OF fields at each time step
        this->updateInputFieldUnsteady();

        printToScreen_ = this->isPrintTime(runTime, printIntervalUnsteady_);

        if (printToScreen_)
        {
            Info << "Time = " << runTime.timeName() << nl << endl;
#include "CourantNo.H"
        }

        // --- Pressure-velocity PIMPLE corrector loop
        while (pimple.loop())
        {
            if (pimple.finalIter() && printToScreen_)
            {
                pimplePrintToScreen = 1;
            }
            else
            {
                pimplePrintToScreen = 0;
            }

#include "UEqnPimple.H"

            // --- Pressure corrector loop
            while (pimple.correct())
            {
#include "pEqnPimple.H"
            }

            if (hasTField_)
            {
#include "TEqnPimple.H"
            }

            laminarTransport.correct();
            daTurbulenceModelPtr_->correct(pimplePrintToScreen);

            // update the output field value at each iteration, if the regression model is active
            fail = daRegressionPtr_->compute();
        }

        regModelFail += fail;

        if (this->validateStates())
        {
            // write data to files and quit
            runTime.writeNow();
            mesh.write();
            return 1;
        }

        this->calcAllFunctions(printToScreen_);
        daRegressionPtr_->printInputInfo(printToScreen_);
        daTurbulenceModelPtr_->printYPlus(printToScreen_);
        this->printElapsedTime(runTime, printToScreen_);

        if (reduceIO && iter < nInstances)
        {
            this->writeAdjStates(reduceIOWriteMesh_, additionalOutput);
            daRegressionPtr_->writeFeatures();
        }
        else
        {
            runTime.write();
            daRegressionPtr_->writeFeatures();
        }
    }

    if (regModelFail != 0)
    {
        return 1;
    }

    // need to save primalFinalTimeIndex_.
    primalFinalTimeIndex_ = runTime.timeIndex();

    // write the mesh to files
    mesh.write();

    Info << "End\n"
         << endl;

    return 0;
}

scalar DAPimpleFoam::calcAdjointResiduals(
    const double* psi,
    const double* dFdW,
    double* adjRes)
{
    scalar adjResNorm = 0.0;
#ifdef CODI_ADR
    label localAdjSize = daIndexPtr_->nLocalAdjointStates;
    // calculate the adjoint residuals  dRdWT*Psi - dFdW
    this->assignVec2ResidualGradient(psi);
    this->globalADTape_.evaluate();
    this->assignStateGradient2Vec(adjRes);
    // NOTE: we do NOT normalize dRdWOldTPsi!
    // this->normalizeGradientVec(adjRes);
    this->globalADTape_.clearAdjoints();
    for (label i = 0; i < localAdjSize; i++)
    {
        adjRes[i] -= dFdW[i];
        adjResNorm += adjRes[i] * adjRes[i];
    }
    reduce(adjResNorm, sumOp<scalar>());
    adjResNorm = sqrt(adjResNorm);
#endif
    return adjResNorm;
}

label DAPimpleFoam::solveAdjointFP(
    Vec dFdW,
    Vec psi)
{
    /*
    Description:
        Solve the adjoint using the fixed-point iteration approach
    */

#ifdef CODI_ADR

    Info << "Solving adjoint using fixed-point iterations" << endl;

    PetscScalar* psiArray;
    PetscScalar* dFdWArray;
    VecGetArray(psi, &psiArray);
    VecGetArray(dFdW, &dFdWArray);

    const fvSolution& myFvSolution = meshPtr_->thisDb().lookupObject<fvSolution>("fvSolution");
    dictionary solverDictU = myFvSolution.subDict("solvers").subDict("U");
    dictionary solverDictP = myFvSolution.subDict("solvers").subDict("p");
    scalar fpRelaxU = daOptionPtr_->getAllOptions().subDict("adjEqnOption").lookupOrDefault<scalar>("fpRelaxU", 1.0);
    scalar fpRelaxP = daOptionPtr_->getAllOptions().subDict("adjEqnOption").lookupOrDefault<scalar>("fpRelaxP", 1.0);
    scalar fpRelaxPhi = daOptionPtr_->getAllOptions().subDict("adjEqnOption").lookupOrDefault<scalar>("fpRelaxPhi", 1.0);
    label fpPrintInterval = daOptionPtr_->getAllOptions().subDict("adjEqnOption").lookupOrDefault<label>("fpPrintInterval", 10);
    label useNonZeroInitGuess = daOptionPtr_->getAllOptions().subDict("adjEqnOption").getLabel("useNonZeroInitGuess");
    label fpMaxIters = daOptionPtr_->getAllOptions().subDict("adjEqnOption").getLabel("fpMaxIters");
    scalar fpRelTol = daOptionPtr_->getAllOptions().subDict("adjEqnOption").getScalar("fpRelTol");

    label localAdjSize = daIndexPtr_->nLocalAdjointStates;
    double* adjRes = new double[localAdjSize];
    for (label i = 0; i < localAdjSize; i++)
    {
        adjRes[i] = 0.0;
        if (!useNonZeroInitGuess)
        {
            psiArray[i] = 0.0;
        }
    }

    volVectorField& U = const_cast<volVectorField&>(
        meshPtr_->thisDb().lookupObject<volVectorField>("U"));

    volScalarField& p = const_cast<volScalarField&>(
        meshPtr_->thisDb().lookupObject<volScalarField>("p"));

    surfaceScalarField& phi = const_cast<surfaceScalarField&>(
        meshPtr_->thisDb().lookupObject<surfaceScalarField>("phi"));

    DATurbulenceModel& daTurb = const_cast<DATurbulenceModel&>(daModelPtr_->getDATurbulenceModel());

    volVectorField dPsiU("dPsiU", U);
    volScalarField dPsiP("dPsiP", p);
    scalarList turbVar(meshPtr_->nCells(), 0.0);
    scalarList dPsiTurbVar(meshPtr_->nCells(), 0.0);

    // NOTE: we need only the fvm operators for the fixed-point preconditioner
    // velocity
    fvVectorMatrix psiUPC(
        fvm::ddt(dPsiU)
        + fvm::div(phi, dPsiU, "div(phi,U)")
        - fvm::laplacian(daTurb.nuEff(), dPsiU));
    psiUPC.relax(1.0);
    // transpose the matrix
    DAUtility::swapLists<scalar>(psiUPC.upper(), psiUPC.lower());
    // make sure the boundary contribution to source hrs is zero
    forAll(psiUPC.boundaryCoeffs(), patchI)
    {
        forAll(psiUPC.boundaryCoeffs()[patchI], faceI)
        {
            psiUPC.boundaryCoeffs()[patchI][faceI] = vector::zero;
        }
    }

    // pressure
    volScalarField rAU(1.0 / psiUPC.A());
    fvScalarMatrix psiPPC(
        fvm::laplacian(rAU, dPsiP));
    // transpose the matrix
    DAUtility::swapLists(psiPPC.upper(), psiPPC.lower());
    // make sure the boundary contribution to source hrs is zero
    forAll(psiPPC.boundaryCoeffs(), patchI)
    {
        forAll(psiPPC.boundaryCoeffs()[patchI], faceI)
        {
            psiPPC.boundaryCoeffs()[patchI][faceI] = 0.0;
        }
    }

    // first, we setup the AD environment for dRdWT*Psi
    this->globalADTape_.reset();
    this->globalADTape_.setActive();

    this->registerStateVariableInput4AD();
    this->updateStateBoundaryConditions();
    this->calcResiduals();
    this->registerResidualOutput4AD();
    this->globalADTape_.setPassive();
    // AD ready to use

    // print the initial residual
    scalar adjResL2Norm0 = this->calcAdjointResiduals(psiArray, dFdWArray, adjRes);
    Info << "Iter: 0. L2 Norm Residual: " << adjResL2Norm0 << ". "
         << runTimePtr_->elapsedCpuTime() << " s" << endl;

    for (label n = 1; n <= fpMaxIters; n++)
    {
        // U
        forAll(dPsiU, cellI)
        {
            for (label comp = 0; comp < 3; comp++)
            {
                label localIdx = daIndexPtr_->getLocalAdjointStateIndex("U", cellI, comp);
                psiUPC.source()[cellI][comp] = adjRes[localIdx];
            }
        }
        forAll(dPsiU, cellI)
        {
            for (label comp = 0; comp < 3; comp++)
            {
                dPsiU[cellI][comp] = 0;
            }
        }
        psiUPC.solve(solverDictU);

        forAll(dPsiU, cellI)
        {
            for (label comp = 0; comp < 3; comp++)
            {
                label localIdx = daIndexPtr_->getLocalAdjointStateIndex("U", cellI, comp);
                psiArray[localIdx] -= dPsiU[cellI][comp].value() * fpRelaxU.value();
            }
        }

        for (label i = 0; i < 2; i++)
        {
            // p
            // update the adjoint residual
            this->calcAdjointResiduals(psiArray, dFdWArray, adjRes);

            forAll(dPsiP, cellI)
            {
                label localIdx = daIndexPtr_->getLocalAdjointStateIndex("p", cellI);
                psiPPC.source()[cellI] = adjRes[localIdx];
            }
            forAll(dPsiP, cellI)
            {
                dPsiP[cellI] = 0;
            }
            psiPPC.solve(solverDictP);

            forAll(dPsiP, cellI)
            {
                label localIdx = daIndexPtr_->getLocalAdjointStateIndex("p", cellI);
                psiArray[localIdx] -= dPsiP[cellI].value() * fpRelaxP.value();
            }
            // update the adjoint residual
            this->calcAdjointResiduals(psiArray, dFdWArray, adjRes);

            // phi
            forAll(meshPtr_->faces(), faceI)
            {
                label localIdx = daIndexPtr_->getLocalAdjointStateIndex("phi", faceI);
                psiArray[localIdx] += adjRes[localIdx] * fpRelaxPhi.value();
            }
        }

        // update the adjoint residual
        this->calcAdjointResiduals(psiArray, dFdWArray, adjRes);

        // turbulence vars
        forAll(stateInfo_["modelStates"], idxI)
        {

            const word turbVarName = stateInfo_["modelStates"][idxI];
            scalar fpRelaxTurbVar = daOptionPtr_->getAllOptions().subDict("adjEqnOption").lookupOrDefault<scalar>("fpRelax" + turbVarName, 1.0);
            forAll(turbVar, cellI)
            {
                label localIdx = daIndexPtr_->getLocalAdjointStateIndex(turbVarName, cellI);
                turbVar[cellI] = adjRes[localIdx];
            }
            daTurbulenceModelPtr_->solveAdjointFP(turbVarName, turbVar, dPsiTurbVar);
            forAll(dPsiTurbVar, cellI)
            {
                label localIdx = daIndexPtr_->getLocalAdjointStateIndex(turbVarName, cellI);
                psiArray[localIdx] -= dPsiTurbVar[cellI].value() * fpRelaxTurbVar.value();
            }
        }

        // update the residual and print to the screen
        scalar adjResL2Norm = this->calcAdjointResiduals(psiArray, dFdWArray, adjRes);
        if (n % fpPrintInterval == 0 || n == fpMaxIters || (adjResL2Norm / adjResL2Norm0) < fpRelTol)
        {
            Info << "Iter: " << n << ". L2 Norm Residual: " << adjResL2Norm << ". "
                 << runTimePtr_->elapsedCpuTime() << " s" << endl;

            if ((adjResL2Norm / adjResL2Norm0) < fpRelTol)
            {
                break;
            }
        }
    }

    // **********************************************************************************************
    // clean up OF vars's AD seeds by deactivating the inputs and call the forward func one more time
    // **********************************************************************************************
    this->deactivateStateVariableInput4AD();
    this->updateStateBoundaryConditions();
    this->calcResiduals();

    delete[] adjRes;
    VecRestoreArray(psi, &psiArray);
    VecRestoreArray(dFdW, &dFdWArray);
#endif

    return 0;
}

} // End namespace Foam

// ************************************************************************* //
