/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v3

    This class is modified from OpenFOAM's source code
    applications/solvers/compressible/rhoPimpleFoam
    NOTE: we use the pimpleFoam implementation from OF-2.4.x

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

#include "DARhoPimpleFoam.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(DARhoPimpleFoam, 0);
addToRunTimeSelectionTable(DASolver, DARhoPimpleFoam, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DARhoPimpleFoam::DARhoPimpleFoam(
    char* argsAll,
    PyObject* pyOptions)
    : DASolver(argsAll, pyOptions),
      pimplePtr_(nullptr),
      pThermoPtr_(nullptr),
      pPtr_(nullptr),
      rhoPtr_(nullptr),
      UPtr_(nullptr),
      phiPtr_(nullptr),
      dpdtPtr_(nullptr),
      KPtr_(nullptr),
      turbulencePtr_(nullptr),
      daTurbulenceModelPtr_(nullptr),
      daFvSourcePtr_(nullptr),
      fvSourcePtr_(nullptr),
      fvSourceEnergyPtr_(nullptr)
{
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

void DARhoPimpleFoam::initSolver()
{
    /*
    Description:
        Initialize variables for DASolver
    */

    Info << "Initializing fields for DARhoPimpleFoam" << endl;
    Time& runTime = runTimePtr_();
    fvMesh& mesh = meshPtr_();
    argList& args = argsPtr_();
#include "createPimpleControlPython.H"
#include "createFieldsRhoPimple.H"
#include "createAdjointCompressible.H"
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

    // reduceIO does not write mesh, but if there is a FFD variable, set writeMesh to 1
    dictionary dvSubDict = daOptionPtr_->getAllOptions().subDict("designVar");
    forAll(dvSubDict.toc(), idxI)
    {
        word dvName = dvSubDict.toc()[idxI];
        if (dvSubDict.subDict(dvName).getWord("designVarType") == "FFD")
        {
            reduceIOWriteMesh_ = 1;
            break;
        }
    }
}

label DARhoPimpleFoam::solvePrimal(
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

#include "createRefsRhoPimple.H"

    // change the run status
    daOptionPtr_->setOption<word>("runStatus", "solvePrimal");

    // we need to read in the states from the 0 folder every time we start the primal
    // here we read in all time levels
    runTime.setTime(0.0, 0);
    this->readStateVars(0.0, 0);
    this->readStateVars(0.0, 1);

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

    label printInterval = daOptionPtr_->getOption<label>("printIntervalUnsteady");
    label printToScreen = 0;
    label pimplePrintToScreen = 0;

    // reset the unsteady obj func to zeros
    this->initUnsteadyObjFuncs();

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

    // check if the parameters are set in the Python layer
    daRegressionPtr_->validate();

    // main loop
    label regModelFail = 0;
    label fail = 0;

    for (label iter = 1; iter <= nInstances; iter++)
    {
        ++runTime;

        printToScreen = this->isPrintTime(runTime, printInterval);

        if (printToScreen)
        {
            Info << "Time = " << runTime.timeName() << nl << endl;
        }

        if (pimple.nCorrPIMPLE() <= 1)
        {
#include "rhoEqnRhoPimple.H"
        }

        // --- Pressure-velocity PIMPLE corrector loop
        while (pimple.loop())
        {

            if (pimple.finalIter() && printToScreen)
            {
                pimplePrintToScreen = 1;
            }
            else
            {
                pimplePrintToScreen = 0;
            }

            // Pressure-velocity SIMPLE corrector
#include "UEqnRhoPimple.H"
#include "EEqnRhoPimple.H"
            // --- Pressure corrector loop
            while (pimple.correct())
            {
#include "pEqnRhoPimple.H"
            }

            // update the output field value at each iteration, if the regression model is active
            fail = daRegressionPtr_->compute();

            daTurbulenceModelPtr_->correct(pimplePrintToScreen);
        }

        regModelFail += fail;

        if (this->validateStates())
        {
            // write data to files and quit
            runTime.writeNow();
            mesh.write();
            return 1;
        }

        this->calcUnsteadyObjFuncs();

        if (printToScreen)
        {
#include "CourantNo.H"
            daTurbulenceModelPtr_->printYPlus();

            this->printAllObjFuncs();

            daRegressionPtr_->printInputInfo();

            Info << "ExecutionTime = " << runTime.elapsedCpuTime() << " s"
                 << "  ClockTime = " << runTime.elapsedClockTime() << " s"
                 << nl << endl;
        }

        if (reduceIO && iter < nInstances)
        {
            this->writeAdjStates(reduceIOWriteMesh_, additionalOutput);
        }
        else
        {
            runTime.write();
        }
    }

    if (regModelFail != 0)
    {
        return 1;
    }

    this->calcPrimalResidualStatistics("print");

    // primal converged, assign the OpenFoam fields to the state vec wVec
    this->ofField2StateVec(wVec);

    // write the mesh to files
    mesh.write();

    Info << "End\n"
         << endl;

    return 0;
}

} // End namespace Foam

// ************************************************************************* //
