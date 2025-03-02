/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

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
      MRFPtr_(nullptr),
      PrPtr_(nullptr),
      PrtPtr_(nullptr),
      TPtr_(nullptr),
      alphatPtr_(nullptr)
{
    // check whether the temperature field exists in the 0 folder
    hasTField_ = DAUtility::isFieldReadable(meshPtr_(), "T", "volScalarField");

    // get fvSolution and fvSchemes info for fixed-point adjoint
    const fvSolution& myFvSolution = meshPtr_->thisDb().lookupObject<fvSolution>("fvSolution");
    if (myFvSolution.found("relaxationFactors"))
    {
        if (myFvSolution.subDict("relaxationFactors").found("equations"))
        {
            if (myFvSolution.subDict("relaxationFactors").subDict("equations").found("U"))
            {
                relaxUEqn_ = myFvSolution.subDict("relaxationFactors").subDict("equations").getScalar("U");
            }
        }
    }
    solverDictU_ = myFvSolution.subDict("solvers").subDict("U");
    solverDictP_ = myFvSolution.subDict("solvers").subDict("p");
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

label DASimpleFoam::solvePrimal()
{
    /*
    Description:
        Call the primal solver to get converged state variables
    */

#include "createRefsSimple.H"
#include "createFvOptions.H"

    // call correctNut, this is equivalent to turbulence->validate();
    daTurbulenceModelPtr_->updateIntermediateVariables();

    Info << "\nStarting time loop\n"
         << endl;

    while (this->loop(runTime)) // using simple.loop() will have seg fault in parallel
    {

        if (printToScreen_)
        {
            Info << "Time = " << runTime.timeName() << nl << endl;
        }

        p.storePrevIter();

        // --- Pressure-velocity SIMPLE corrector
        {
#include "UEqnSimple.H"
#include "pEqnSimple.H"
            if (hasTField_)
            {
#include "TEqnSimple.H"
            }
        }

        laminarTransport.correct();
        daTurbulenceModelPtr_->correct(printToScreen_);

        // calculate all functions
        this->calcAllFunctions(printToScreen_);
        // calculate yPlus
        daTurbulenceModelPtr_->printYPlus(printToScreen_);
        // compute the regression model and print the feature
        regModelFail_ = daRegressionPtr_->compute();
        daRegressionPtr_->printInputInfo(printToScreen_);
        // print run time
        this->printElapsedTime(runTime, printToScreen_);

        runTime.write();
    }

    // write the mesh to files
    mesh.write();

    // write associated fields such as URel
    this->writeAssociatedFields();

    Info << "End\n"
         << endl;

    return this->checkPrimalFailure();
}

// ************ the following are functions for consistent fixed-point adjoint

label DASimpleFoam::runFPAdj(
    Vec dFdW,
    Vec psi)
{
    // If adjoint converged, then adjConv = 0
    // Otherwise, adjConv = 1
    label adjConv = 1;

#ifdef CODI_ADR
    /*
    Description:
        Solve the adjoint using the fixed-point iteration method
    
    xvVec: the volume mesh coordinate vector

    wVec: the state variable vector

    dFdW: The dF/dW vector     

    psi: The adjoint solution vector
    */

    // Here we keep the values of the previous step psi
    //VecZeroEntries(psi);

    label printInterval = daOptionPtr_->getOption<label>("printInterval");

    word adjEqnSolMethod = daOptionPtr_->getOption<word>("adjEqnSolMethod");

    if (adjEqnSolMethod == "fixedPoint")
    {
        Info << "Solving the adjoint using consistent fixed-point iteration method..."
             << "  Execution Time: " << meshPtr_->time().elapsedCpuTime() << " s" << endl;
        ;
        label fpMaxIters = daOptionPtr_->getSubDictOption<label>("adjEqnOption", "fpMaxIters");
        scalar fpRelTol = daOptionPtr_->getSubDictOption<scalar>("adjEqnOption", "fpRelTol");
        scalar fpMinResTolDiff = daOptionPtr_->getSubDictOption<scalar>("adjEqnOption", "fpMinResTolDiff");

        const objectRegistry& db = meshPtr_->thisDb();
        volVectorField& U = const_cast<volVectorField&>(db.lookupObject<volVectorField>("U"));
        volScalarField& p = const_cast<volScalarField&>(db.lookupObject<volScalarField>("p"));
        surfaceScalarField& phi = const_cast<surfaceScalarField&>(db.lookupObject<surfaceScalarField>("phi"));
        volScalarField& nuTilda = const_cast<volScalarField&>(db.lookupObject<volScalarField>("nuTilda"));
        // nuEff is only used to construct pseudoUEqn
        volScalarField nuEff = daTurbulenceModelPtr_->nuEff();

        // We now pass dFdW directly into adjRes components at the beginning of adjoint residual calculation
        /*
        // Initialize field cmpts for dFdW as all-zero
        volVectorField dFdU("dFdU", 0.0 * U);
        volScalarField dFdP("dFdP", 0.0 * p);
        surfaceScalarField dFdPhi("dFdPhi", 0.0 * phi);
        volScalarField dFdNuTilda("dFdNuTilda", 0.0 * nuTilda);
        */

        // Initialize primal residuals as all-zero
        volVectorField URes("URes", 0.0 * U);
        volScalarField pRes("pRes", 0.0 * p);
        surfaceScalarField phiRes("phiRes", 0.0 * phi);
        volScalarField nuTildaRes("nuTildaRes", 0.0 * nuTilda);

        // We now pass dFdW directly into adjRes components at the beginning of adjoint residual calculation
        /*
        // Pass the values in dFdW to its field cmpts
        this->vec2Fields("vec2Field", dFdW, dFdU, dFdP, dFdPhi, dFdNuTilda);
        */
        /*
        // Flip the sign for now to avoid the strange issue
        dFdU = -dFdU;
        dFdP = -dFdP;
        dFdPhi = -dFdPhi;
        dFdNuTilda = -dFdNuTilda;
        */

        // Initialize all-zero cmpts for the adjoint vector
        volVectorField UPsi("UPsi", 0.0 * U);
        volScalarField pPsi("pPsi", 0.0 * p);
        surfaceScalarField phiPsi("phiPsi", 0.0 * phi);
        volScalarField nuTildaPsi("nuTildaPsi", 0.0 * nuTilda);

        // Pass psiVec to cmpts of the adjoint vector:
        //this->vec2Fields("vec2Field", psi, UPsi, pPsi, phiPsi, nuTildaPsi);

        /*
        // Change every element in the adjoint vector to one
        forAll(UPsi, cellI)
        {
            UPsi[cellI][0] = 1.0;
            UPsi[cellI][1] = 1.0;
            UPsi[cellI][2] = 1.0;
        }
        forAll(pPsi, cellI)
        {
            pPsi[cellI] = 1.0;
        }
        forAll(phiPsi.primitiveFieldRef(), faceI)
        {
            phiPsi.primitiveFieldRef()[faceI] = 1.0;
        }
        forAll(phiPsi.boundaryFieldRef(), patchI)
        {
            forAll(phiPsi.boundaryFieldRef()[patchI], faceI)
            {
                phiPsi.boundaryFieldRef()[patchI][faceI] = 1.0;
            }
        }
        forAll(nuTildaPsi, cellI)
        {
            nuTildaPsi[cellI] = 1.0;
        }
        */

        // Initialize all-zero cmpts for the adjoint residual
        // Note: we tried defining below as alias of pseudo variables, it works but it actually increases mem usage..
        volVectorField adjURes("adjURes", 0.0 * U);
        volScalarField adjPRes("adjPRes", 0.0 * p);
        surfaceScalarField adjPhiRes("adjPhiRes", 0.0 * phi);
        volScalarField adjNuTildaRes("adjNuTildaRes", 0.0 * nuTilda);

        // Initialize the initial L2norm for adjURes, adjPRes, adjPhiRes, adjNuTildaRes as all-zero
        vector initNormAdjURes = vector::zero;
        scalar initNormAdjPRes = 0.0;
        scalar initNormAdjPhiRes = 0.0;
        scalar initNormAdjNuTildaRes = 0.0;

        // Below use the norm of dFdW as the initial residual norm
        /*
        // Pass -dfdw to adjRes:
        this->vec2Fields("vec2Field", dFdW, adjURes, adjPRes, adjPhiRes, adjNuTildaRes);
        adjURes = -adjURes;
        adjPRes = -adjPRes;
        adjPhiRes = -adjPhiRes;
        adjNuTildaRes = -adjNuTildaRes;

        // Initialize the initial L2norm for adjURes, adjPRes, adjPhiRes, adjNuTildaRes
        vector initNormAdjURes = L2norm(adjURes);
        //scalar initNormAdjPRes = L2norm(adjPRes);
        //scalar initNormAdjPhiRes = L2norm(adjPhiRes);
        //scalar initNormAdjNuTildaRes = L2norm(adjNuTildaRes);

        // Get the norm of the total adjoint residual
        scalar initNormTotAdjRes = sqr(initNormAdjURes[0]) +  sqr(initNormAdjURes[1]) + sqr(initNormAdjURes[2]) + sqr(L2norm(adjPRes)) + sqr(L2norm(adjPhiRes)) + sqr(L2norm(adjNuTildaRes));
        initNormTotAdjRes = sqrt(initNormTotAdjRes);
        */

// Construct the pseudoEqns using pseudoEqns.H
// Better to make pseudoUEqn, pseudoPEqn, pseudoNuTildaEqn class variables
// Also need to move pseudoNuTildaEqn to DASpalartAllmarasFv3 under DATurbulenceModel
#include "pseudoEqns.H"

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

        // No need to print the un-normalized primal residuals
        /*
        // Print primal residuals before starting the adjoint iterations
        this->calcLduResiduals(URes, pRes, phiRes);
        daTurbulenceModelPtr_->calcLduResidualTurb(nuTildaRes);
        Info << "Residual for simpleFOAM after convergence: " << endl;
        Info << "L2 norm of URes: " << L2norm(URes) << endl;
        Info << "L2 norm of pRes: " << L2norm(pRes) << endl;
        Info << "L2 norm of phiRes: " << L2norm(phiRes) << endl;
        Info << "L2 norm of nuTildaRes: " << L2norm(nuTildaRes) << endl;
        */

        label cnt = 0;
        while (cnt < fpMaxIters)
        {
            if (cnt % printInterval == 0)
            {
                Info << "Step = " << cnt << "  Execution Time: " << meshPtr_->time().elapsedCpuTime() << " s" << endl;
            }

            // ************************************************************************* //
            // Step-1: Get D^(-T).adjURes, adjPRes, adjPhiRes, adjNuTildaRes, and update phiPsi
            // ************************************************************************* //
            // Step-1.1: Calculate adjURes, adjPRes, adjPhiRes, adjNuTildaRes, and update phiPsi

            // At beginning of iteration, call calcAdjointResidual once to get the adjoint residual
            this->calcAdjointResidual(URes, pRes, phiRes, nuTildaRes, dFdW, UPsi, pPsi, phiPsi, nuTildaPsi, adjURes, adjPRes, adjPhiRes, adjNuTildaRes, cnt);

            // Below uses the norm of total adjoint residual to check convergence
            /*
            // Calculate L2 norm for all components of adjoint residual, then check if fpRelTol is met
            vector normAdjURes = L2norm(adjURes);
            scalar normAdjPRes = L2norm(adjPRes);
            scalar normAdjPhiRes = L2norm(adjPhiRes);
            scalar normAdjNuTildaRes = L2norm(adjNuTildaRes);

            scalar normTotAdjRes = sqr(normAdjURes[0]) +  sqr(normAdjURes[1]) + sqr(normAdjURes[2]) + sqr(normAdjPRes) + sqr(normAdjPhiRes) + sqr(normAdjNuTildaRes);
            normTotAdjRes = sqrt(normTotAdjRes);

            // Normalize total residual norm:
            normTotAdjRes /= initNormTotAdjRes;

            if (cnt % printInterval == 0)
            {
                Info << "L2 norm of adjURes: " << normAdjURes[0] << " " << normAdjURes[1] << " " << normAdjURes[2] << endl;
                Info << "L2 norm of adjPRes: " << normAdjPRes << endl;
                Info << "L2 norm of adjPhiRes: " << normAdjPhiRes << endl;
                Info << "L2 norm of adjNuTildaRes: " << normAdjNuTildaRes << endl;
            }

            // Check if fpRelTol is met:
            if (normTotAdjRes < fpRelTol)
            {
                Info << "Residual drop of " << fpRelTol << " has been achieved after " << cnt << " steps!  Execution Time: " << meshPtr_->time().elapsedCpuTime() << " s" << endl;;
                Info << "Final L2 norm of adjURes: " << normAdjURes[0] << " " << normAdjURes[1] << " " << normAdjURes[2] << endl;
                Info << "Final L2 norm of adjPRes: " << normAdjPRes << endl;
                Info << "Final L2 norm of adjPhiRes: " << normAdjPhiRes << endl;
                Info << "Final L2 norm of adjNuTildaRes: " << normAdjNuTildaRes << endl;
                break;
            }
            */

            // Calculate L2 norm for all components of adjoint residual, then check if fpRelTol is met
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

                if (cnt % printInterval == 0)
                {
                    Info << "Normalized L2 norm of adjURes: " << normAdjURes[0] << " " << normAdjURes[1] << " " << normAdjURes[2] << endl;
                    Info << "Normalized L2 norm of adjPRes: " << normAdjPRes << endl;
                    Info << "Normalized L2 norm of adjPhiRes: " << normAdjPhiRes << endl;
                    Info << "Normalized L2 norm of adjNuTildaRes: " << normAdjNuTildaRes << endl;
                }

                // Check if fpRelTol is met:
                if (normAdjURes[0] < fpRelTol && normAdjURes[1] < fpRelTol && normAdjURes[2] < fpRelTol && normAdjPRes < fpRelTol && normAdjPhiRes < fpRelTol && normAdjNuTildaRes < fpRelTol)
                {
                    Info << "Residual drop of " << fpRelTol << " has been achieved after " << cnt << " steps!  Execution Time: " << meshPtr_->time().elapsedCpuTime() << " s" << endl;
                    Info << "Adjoint is converged and succussful.." << endl;
                    Info << "Normalized L2 norm of adjURes: " << normAdjURes[0] << " " << normAdjURes[1] << " " << normAdjURes[2] << endl;
                    Info << "Normalized L2 norm of adjPRes: " << normAdjPRes << endl;
                    Info << "Normalized L2 norm of adjPhiRes: " << normAdjPhiRes << endl;
                    Info << "Normalized L2 norm of adjNuTildaRes: " << normAdjNuTildaRes << endl;
                    adjConv = 0;
                    break;
                }
            }

            // Update phiPsi
            phiPsi += adjPhiRes;

            // ************************************************************************* //
            // Step-1.2: Get D^(-T) adjURes, overwrite adjURes

            forAll(adjURes, cellI)
            {
                adjURes[cellI] /= UDiag[cellI];
            }

            // ************************************************************************* //
            // Step-2: Get alphaP.(F_p^grad)^T.(D^(-T).adjURes) - alphaP.adjPRes + (F_p^flux.F_p^aug)^T.adjPhiRes
            // ************************************************************************* //
            // Step-2.1: Start with (F_p^grad)^T.(D^(-T).adjURes) - adjPRes

            // Initialize step_2 as -adjPRes
            //volScalarField step_2("step_2", -adjPRes);
            // Use alias to save some mem usage
            volScalarField& step_2 = adjPRes;
            step_2 = -step_2;

            // Do (F_p^grad)^T.(D^(-T).adjURes) with ADR
            // Here, F_p^grad is the operator of fvc::grad(p) * mesh().V() on p

            if (cnt == 0)
            {
                this->globalADTape_.setActive();

                // Now register p1 as input:
                forAll(p1, cellI)
                {
                    this->globalADTape_.registerInput(p1[cellI]);
                }

                // Get tape position before calling function(s)
                gradPStart_ = this->globalADTape_.getPosition();

                // Correct boundaries to link the intermediate results
                p1.correctBoundaryConditions();
                U.correctBoundaryConditions();

                // Calculate fvc::grad(p) * mesh().V()
                gradP = fvc::grad(p1);
                forAll(gradP, cellI)
                {
                    gradP.primitiveFieldRef()[cellI] *= U.mesh().V()[cellI];
                }

                // Get tape position before calling function(s)
                gradPEnd_ = this->globalADTape_.getPosition();

                // stop recording
                this->globalADTape_.setPassive();
            }

            // set the AD seed to the output var
            // Seed with (D^(-T).adjURes), which is the overwritten adjURes.
            forAll(gradP, cellI)
            {
                gradP[cellI][0].setGradient(adjURes[cellI][0].getValue());
                gradP[cellI][1].setGradient(adjURes[cellI][1].getValue());
                gradP[cellI][2].setGradient(adjURes[cellI][2].getValue());
            }

            // evaluate the tape to compute the derivative of the seeded output wrt all the input
            this->globalADTape_.evaluate(gradPEnd_, gradPStart_);
            forAll(p1, cellI)
            {
                step_2[cellI] += p1[cellI].getGradient();
            }

            // Clear adjoints for future Jacobian calculations
            this->globalADTape_.clearAdjoints();

            // ************************************************************************* //
            // Step-2.2: Get alphaP*step_2.1 =  alphaP.((F_p^grad)^T.(D^(-T).adjURes) - adjPRes)
            scalar alphaP = p.mesh().fieldRelaxationFactor(p.name());
            step_2 *= alphaP;

            //Info<<"Show step_2 after *alphaP: "<< step_2 << endl;

            // ************************************************************************* //
            // Step-2.3: Get step_2.2 + (F_p^flux.F_p^aug)^T.adjPhiRes  =  alphaP.((F_p^grad)^T.(D(-T).adjURes) - adjPRes) + (F_p^flux.F_p^aug)^T.adjPhiRes
            // Here, (F_p^flux.F_p^aug) is the operator pEqn.flux() on p, (F_p^flux.F_p^aug)^T.adjPhiRes is calculated with ADR

            if (cnt == 0)
            {
                // First, get pEqn1, note that we only need the l.h.s. of pEqn1
                fvScalarMatrix pEqn1(
                    fvm::laplacian(rAU, p2));
                //Info<<"Show pEqn1.flux(): "<< pEqn1.flux() << endl;

                // Re-active the tape, no need to initiate it again
                this->globalADTape_.setActive();

                // Now register p2 as input:
                forAll(p2, cellI)
                {
                    this->globalADTape_.registerInput(p2[cellI]);
                }

                // Get tape position before calling function(s)
                pEqnFluxStart_ = this->globalADTape_.getPosition();

                // Correct boundaries to link the intermediate results
                p2.correctBoundaryConditions();
                U.correctBoundaryConditions();

                // Calculate pEqn1.flux():
                pEqnFlux = pEqn1.flux();

                // Get tape position after calling function(s)
                pEqnFluxEnd_ = this->globalADTape_.getPosition();

                // stop recording
                this->globalADTape_.setPassive();
            }

            // set the AD seed to the output var pEqnFlux
            // Seed with adjPhiRes
            // Seed internal of pEqnFlux:
            forAll(pEqnFlux.primitiveFieldRef(), faceI)
            {
                pEqnFlux.primitiveFieldRef()[faceI].setGradient(adjPhiRes.primitiveFieldRef()[faceI].getValue());
            }
            // Seed boundary of pEqnFlux:
            forAll(pEqnFlux.boundaryFieldRef(), patchI)
            {
                forAll(pEqnFlux.boundaryFieldRef()[patchI], faceI)
                {
                    pEqnFlux.boundaryFieldRef()[patchI][faceI].setGradient(adjPhiRes.boundaryFieldRef()[patchI][faceI].getValue());
                }
            }

            // evaluate the tape to compute the derivative of the seeded output wrt all the input
            this->globalADTape_.evaluate(pEqnFluxEnd_, pEqnFluxStart_);
            forAll(p2, cellI)
            {
                step_2[cellI] += p2[cellI].getGradient();
            }

            // Clear adjoints for future Jacobian calculations
            this->globalADTape_.clearAdjoints();
            // reset the tape for future recording
            //tape.reset();

            //Info<<"Show final form of step_2: "<< step_2 << endl;

            // ************************************************************************* //
            // Step-3: Solve and get step_3 = A_p^(-T).step_2,
            // which is A_p^(-T).(alphaP.((F_p^grad)^T.(D(-T).adjURes) - adjPRes) + (F_p^flux.F_p^aug)^T.adjPhiRes)
            // then update pPsi += step_3

            // Solve, using the inverse transpose product function
            //invTranProd_pEqn(rAU, p, step_2, pseudoP);
            //Info<<"Show pseudoP: "<< pseudoP << endl;

            // *********************************** //
            // Below is equivalent to: invTranProd_pEqn(rAU, p, step_2, pseudoP);
            // Here, pSource is a redundant alias, which is an "input argument" that takes the value of step_2 into solvePseudopEqn.H
            volScalarField& pSource = step_2;
#include "solvePseudoPEqn.H"
            // *********************************** //

            // Get step_3 as the alias of pseudoP
            volScalarField& step_3 = pseudoP;
            //Info<<"Show step_3: "<< step_3 << endl;

            // Update pPsi
            pPsi += step_3;
            //Info<<"Show pPsi: "<< pPsi << endl;

            // ************************************************************************* //
            // Step-4: Get to the point before solve with A_U^(-T)
            // which is Q^T.{D^(-T).adjURes - D^(-T)[E^T.F^T.M_p^(-T).alphaP.B^T.D^(-T).adjURes - E^T.F^T.M_p^(-T).alphaP.adjPRes + (E^T.F^T.M_p^(-T).G^T - E^T).adjPhiRes]}
            // ************************************************************************* //
            // Step-4.1: Get (F_phi^div)^T.step_3 - adjPhiRes = (F_phi^div)^T.(A_p^(-T).(alphaP.((F_p^grad)^T.(D(-T).adjURes) - adjPRes) + (F_p^flux.F_p^aug)^T.adjPhiRes)) - adjPhiRes
            // (F_phi^div)^T.step_3 is calculated with ADR
            // Here, (F_phi^div) is the operator fvc::div(phiHbyA) on phiHbyA
            // We don't want to calculate phiHbyA, so let's use phi instead
            // So, (F_phi^div) is the operator fvc::div(phi) on phi

            // Initiate a phi-like temp variable as - adjPhiRes
            //surfaceScalarField phiTemp = -adjPhiRes;
            // Use alia to same some mem usage
            surfaceScalarField& phiTemp = adjPhiRes;
            phiTemp = -phiTemp;

            if (cnt == 0)
            {
                // Re-active the tape, no need to initiate it again
                this->globalADTape_.setActive();

                // Register phi1 as input
                // Note that both the internal field of phi1 and the boundary fields need to be registered
                // Register internal of phi1:
                forAll(phi1.primitiveFieldRef(), faceI)
                {
                    this->globalADTape_.registerInput(phi1.primitiveFieldRef()[faceI]);
                }
                // Register boundary of phi:
                forAll(phi1.boundaryFieldRef(), patchI)
                {
                    forAll(phi1.boundaryFieldRef()[patchI], faceI)
                    {
                        this->globalADTape_.registerInput(phi1.boundaryFieldRef()[patchI][faceI]);
                    }
                }

                // Get tape position before calling function(s)
                divPhiStart_ = this->globalADTape_.getPosition();

                // This is not needed, but whatever... XD
                // Correct boundaries to link the intermediate results
                //U.correctBoundaryConditions();
                //p.correctBoundaryConditions();

                // Calculate fvc::div(phi), then scale with cell volumes mesh().V()
                divPhi = fvc::div(phi1);
                forAll(divPhi, cellI)
                {
                    divPhi.primitiveFieldRef()[cellI] *= p.mesh().V()[cellI];
                }

                // Get tape position after calling function(s)
                divPhiEnd_ = this->globalADTape_.getPosition();

                // stop recording
                this->globalADTape_.setPassive();
            }

            // set the AD seed to the output var divPhi
            // Seed with step_3
            forAll(divPhi, cellI)
            {
                divPhi[cellI].setGradient(step_3[cellI].getValue());
            }

            // evaluate the tape to compute the derivative of the seeded output wrt all the input
            this->globalADTape_.evaluate(divPhiEnd_, divPhiStart_);
            forAll(phi1.primitiveFieldRef(), faceI)
            {
                phiTemp.primitiveFieldRef()[faceI] += phi1.primitiveFieldRef()[faceI].getGradient();
            }
            forAll(phi1.boundaryFieldRef(), patchI)
            {
                forAll(phi1.boundaryFieldRef()[patchI], faceI)
                {
                    phiTemp.boundaryFieldRef()[patchI][faceI] += phi1.boundaryFieldRef()[patchI][faceI].getGradient();
                }
            }

            // Clear adjoints for future Jacobian calculations
            this->globalADTape_.clearAdjoints();
            // reset the tape for future recording
            //tape.reset();

            //Info<<"Show step-4.1: "<< phiTemp << endl;

            // ************************************************************************* //
            // Step-4.2: Get (D^(-T).adjURes) - D^(-T).(F_U^flux.F_U^aug)^T.phiTemp
            // Here, (F_U^flux.F_U^aug) is the operator fvc::flux(HbyA) on HbyA
            // We don't want to calculate HbyA, so we use U instead
            // HbyA and U have the same boundary settings, so the operator as a function stays the same
            // So, (F_U^flux.F_U^aug) is the operator fvc::flux(U) on U

            // Update UPsi, for the 1st time
            // We have to do this before creating step_4 as the alias of adjURes
            forAll(UPsi, cellI)
            {
                UPsi[cellI] -= adjURes[cellI]; // Note that the overwritten adjURes is actually D^(-T).adjURes
            }

            // Initiate a U-like temporary variable step_4 as D^(-T).adjURes, which is the overwritten adjURes
            //volVectorField step_4("step_4", adjURes);
            // Use alias to save some mem usage
            volVectorField& step_4 = adjURes;

            if (cnt == 0)
            {
                // Re-active the tape, no need to initiate it again
                this->globalADTape_.setActive();

                // Register U1 as input, note that U1 has 3 components
                forAll(U1, cellI)
                {
                    this->globalADTape_.registerInput(U1[cellI][0]);
                    this->globalADTape_.registerInput(U1[cellI][1]);
                    this->globalADTape_.registerInput(U1[cellI][2]);
                }

                // Get tape position before calling function(s)
                fluxUStart_ = this->globalADTape_.getPosition();

                // Correct boundaries to link the intermediate results
                U1.correctBoundaryConditions();
                p.correctBoundaryConditions();

                // Calculate fvc::flux(U)
                fluxU = fvc::flux(U1);

                // Get tape position after calling function(s)
                fluxUEnd_ = this->globalADTape_.getPosition();

                // stop recording
                this->globalADTape_.setPassive();
            }

            // set the AD seed to the output var fluxU
            // Seed with phiTemp
            // Seed internal of fluxU:
            forAll(fluxU.primitiveFieldRef(), faceI)
            {
                fluxU.primitiveFieldRef()[faceI].setGradient(phiTemp.primitiveFieldRef()[faceI].getValue());
            }
            // Seed boundary of fluxU:
            forAll(fluxU.boundaryFieldRef(), patchI)
            {
                forAll(fluxU.boundaryFieldRef()[patchI], faceI)
                {
                    fluxU.boundaryFieldRef()[patchI][faceI].setGradient(phiTemp.boundaryFieldRef()[patchI][faceI].getValue());
                }
            }

            // evaluate the tape to compute the derivative of the seeded output wrt all the input
            this->globalADTape_.evaluate(fluxUEnd_, fluxUStart_);
            forAll(U1, cellI)
            {
                for (label cmpt = 0; cmpt < 3; cmpt++)
                {
                    step_4[cellI][cmpt] -= U1[cellI][cmpt].getGradient() / UDiag[cellI];
                }
            }

            // Clear adjoints for future Jacobian calculations
            this->globalADTape_.clearAdjoints();
            // reset the tape for future recording
            //tape.reset();

            //Info<<"Show step_4: "<< step_4 << endl;

            // ************************************************************************* //
            // Step-4.3: Get Q^T.step_4, this is the last sub-step of step-4
            // We use pseudoUEqn, whose upper and lower are already swapped
            step_4.primitiveFieldRef() = -pseudoUEqn.lduMatrix::H(step_4);
            //Info<<"Show step_4: "<< step_4 << endl;

            // ************************************************************************* //
            // Step-5: Solve and get step_5 = A_U^(-T).step_4
            // Then update UPsi += (step_5 - D^(-T).adjURes)

            // Solve, using the inverse transpose product function
            //invTranProd_UEqn(U, phi, nuEff, p, step_4, pseudoU);
            //Info<<"Show pseudoU: "<< pseudoU << endl;

            // *********************************** //
            // Below is equivalent to: invTranProd_UEqn(U, phi, nuEff, p, step_4, pseudoU);
            // Here, USource is a redundant alias, which is an "input argument" that takes the value of step_4 into solvePseudoUEqn.H
            volVectorField& USource = step_4;
#include "solvePseudoUEqn.H"
            // *********************************** //

            // Get step_5 as the alias of pseudoU
            volVectorField& step_5 = pseudoU;
            //Info<<"Show step_5: "<< step_5 << endl;

            // Update UPsi, for the 2nd time
            forAll(UPsi, cellI)
            {
                UPsi[cellI] += step_5[cellI];
            }
            //Info<<"Show UPsi: "<< UPsi << endl;

            // ************************************************************************* //
            // Step-6: (SA turbulence) Solve and get pseudoNuTilda = A_nuTilda^(-T).adjNuTildaRes
            // Then update nuTildaPsi -= pseudoNuTilda

            // Solve, using the inverse transpose product function
            //invTranProd_nuTildaEqn(U, phi, nuTilda, y, nu, sigmaNut, kappa, Cb1, Cb2, Cw1, Cw2, Cw3, Cv1, Cv2, adjNuTildaRes, pseudoNuTilda);

            // *********************************** //
            // Below is equivalent to: invTranProd_nuTildaEqn(U, phi, nuTilda, y, nu, sigmaNut, kappa, Cb1, Cb2, Cw1, Cw2, Cw3, Cv1, Cv2, adjNuTildaRes, pseudoNuTilda);
            // Here, nuTildaSource is a redundant alias, which is an "input argument" that takes the value of adjNuTildaRes into solvePseudoUEqn.H

            volScalarField& nuTildaSource = adjNuTildaRes;

            daTurbulenceModelPtr_->rhsSolvePseudoNuTildaEqn(nuTildaSource);
            //#include "solvePseudonuTildaEqn.H"
            // *********************************** //

            // Update nuTildaPsi
            forAll(nuTildaPsi, cellI)
            {
                nuTildaPsi[cellI] -= pseudoNuTilda[cellI];
            }

            cnt++;
        }

        // If fpRelTol not met, check if the relaxed adjoint relTol is met
        if (adjConv == 1)
        {
            Info << "Prescribed adjoint convergence of " << fpRelTol << " has not been met after " << cnt << " steps!  Execution Time: " << meshPtr_->time().elapsedCpuTime() << " s" << endl;
            Info << "Now checking if the relaxed convergence is met..." << endl;
            this->calcAdjointResidual(URes, pRes, phiRes, nuTildaRes, dFdW, UPsi, pPsi, phiPsi, nuTildaPsi, adjURes, adjPRes, adjPhiRes, adjNuTildaRes, cnt);

            vector normAdjURes = L2norm(adjURes);
            scalar normAdjPRes = L2norm(adjPRes);
            scalar normAdjPhiRes = L2norm(adjPhiRes);
            scalar normAdjNuTildaRes = L2norm(adjNuTildaRes);

            // Normalize residual norms:
            normAdjURes[0] /= initNormAdjURes[0];
            normAdjURes[1] /= initNormAdjURes[1];
            normAdjURes[2] /= initNormAdjURes[2];
            normAdjPRes /= initNormAdjPRes;
            normAdjPhiRes /= initNormAdjPhiRes;
            normAdjNuTildaRes /= initNormAdjNuTildaRes;

            scalar relaxedFpRelTol = fpRelTol * fpMinResTolDiff;
            if (normAdjURes[0] < relaxedFpRelTol && normAdjURes[1] < relaxedFpRelTol && normAdjURes[2] < relaxedFpRelTol && normAdjPRes < relaxedFpRelTol && normAdjPhiRes < relaxedFpRelTol && normAdjNuTildaRes < relaxedFpRelTol)
            {
                Info << "Relaxed residual drop of " << relaxedFpRelTol << " has been achieved!  Execution Time: " << meshPtr_->time().elapsedCpuTime() << " s" << endl;
                Info << "Adjoint is still considered succussful.." << endl;
                adjConv = 0;
            }
            else
            {
                Info << "Relaxed residual drop of " << relaxedFpRelTol << " has not been met!  Execution Time: " << meshPtr_->time().elapsedCpuTime() << " s" << endl;
                Info << "Adjoint has failed to converge.." << endl;
            }
            Info << "Normalized L2 norm of adjURes: " << normAdjURes[0] << " " << normAdjURes[1] << " " << normAdjURes[2] << endl;
            Info << "Normalized L2 norm of adjPRes: " << normAdjPRes << endl;
            Info << "Normalized L2 norm of adjPhiRes: " << normAdjPhiRes << endl;
            Info << "Normalized L2 norm of adjNuTildaRes: " << normAdjNuTildaRes << endl;
        }

        // converged, assign the field Psi to psiVec
        this->vec2Fields("field2Vec", psi, UPsi, pPsi, phiPsi, nuTildaPsi);
    }
    else
    {
        FatalErrorIn("adjEqnSolMethod not valid") << exit(FatalError);
    }

#endif
    return adjConv;
}

void DASimpleFoam::vec2Fields(
    const word mode,
    Vec cVec,
    volVectorField& UField,
    volScalarField& pField,
    surfaceScalarField& phiField,
    volScalarField& nuTildaField)
{
#ifdef CODI_ADR
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

    /*
    const objectRegistry& db = meshPtr_->thisDb();
    const surfaceScalarField& phi = db.lookupObject<surfaceScalarField>("phi");
    volScalarField nuEff = daTurbulenceModelPtr_->nuEff();

    // Get the pseudoUEqn,
    // the most important thing here is to make sure the l.h.s. matches that of UEqn.
    fvVectorMatrix pseudoUEqn(
        fvm::div(phi, pseudoU, "div(phi,U)")
        - fvm::laplacian(nuEff, pseudoU)
        - fvc::div(nuEff * dev2(T(fvc::grad(pseudoU))), "div((nuEff*dev2(T(grad(U)))))"));
    pseudoUEqn.relax(relaxUEqn_);

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

    pseudoUEqn.solve(solverDictU_);
*/
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

    /*
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

    pseudoPEqn.solve(solverDictP_);
*/
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
    Vec dFdW,
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
#ifdef CODI_ADR
    volVectorField& U = const_cast<volVectorField&>(meshPtr_->thisDb().lookupObject<volVectorField>("U"));
    volScalarField& p = const_cast<volScalarField&>(meshPtr_->thisDb().lookupObject<volScalarField>("p"));
    volScalarField& nuTilda = const_cast<volScalarField&>(meshPtr_->thisDb().lookupObject<volScalarField>("nuTilda"));
    surfaceScalarField& phi = const_cast<surfaceScalarField&>(meshPtr_->thisDb().lookupObject<surfaceScalarField>("phi"));

    // Pass -dfdw to adjRes:
    this->vec2Fields("vec2Field", dFdW, adjURes, adjPRes, adjPhiRes, adjNuTildaRes);
    adjURes = -adjURes;
    adjPRes = -adjPRes;
    adjPhiRes = -adjPhiRes;
    adjNuTildaRes = -adjNuTildaRes;

    if (cnt == 0)
    {
        this->globalADTape_.reset();
        this->globalADTape_.setActive();

        // register all (3+1) state variables as input
        // Start with U, note that U has 3 components
        forAll(U, cellI)
        {
            this->globalADTape_.registerInput(U[cellI][0]);
            this->globalADTape_.registerInput(U[cellI][1]);
            this->globalADTape_.registerInput(U[cellI][2]);
        }
        // Now register p as input:
        forAll(p, cellI)
        {
            this->globalADTape_.registerInput(p[cellI]);
        }
        // Then, register phi as input
        // Note that both the internal field of phi and the boundary fields need to be registered
        // Register internal of phi:
        forAll(phi.primitiveFieldRef(), faceI)
        {
            this->globalADTape_.registerInput(phi.primitiveFieldRef()[faceI]);
        }
        // Register boundary of phi:
        forAll(phi.boundaryFieldRef(), patchI)
        {
            forAll(phi.boundaryFieldRef()[patchI], faceI)
            {
                this->globalADTape_.registerInput(phi.boundaryFieldRef()[patchI][faceI]);
            }
        }
        // And then, register turbulence variable nuTilda as input:
        forAll(nuTilda, cellI)
        {
            this->globalADTape_.registerInput(nuTilda[cellI]);
        }

        // Get tape position before calling function(s)
        adjResStart_ = this->globalADTape_.getPosition();

        // Correct boundaries to link the intermediate results
        U.correctBoundaryConditions();
        p.correctBoundaryConditions();
        nuTilda.correctBoundaryConditions();

        // Construct nuEff before calling lduCalcAllRes
        daTurbulenceModelPtr_->updateIntermediateVariables();

        // Call the residual functions
        this->calcLduResiduals(URes, pRes, phiRes);
        daTurbulenceModelPtr_->calcLduResidualTurb(nuTildaRes);

        // No registerOutput needed in positional tapes
        /*
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
        */

        // Get tape position after calling function(s)
        adjResEnd_ = this->globalADTape_.getPosition();

        // stop recording
        this->globalADTape_.setPassive();
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
    this->globalADTape_.evaluate(adjResEnd_, adjResStart_);
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
    this->globalADTape_.clearAdjoints();
#endif
}

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

    forAll(Phi.primitiveField(), faceI)
    {
        L2normPhi += sqr(Phi.primitiveField()[faceI]);
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

} // End namespace Foam

// ************************************************************************* //
