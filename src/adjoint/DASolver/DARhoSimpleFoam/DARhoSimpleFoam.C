/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

    This class is modified from OpenFOAM's source code
    applications/solvers/compressible/rhoSimpleFoam

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

#include "DARhoSimpleFoam.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(DARhoSimpleFoam, 0);
addToRunTimeSelectionTable(DASolver, DARhoSimpleFoam, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DARhoSimpleFoam::DARhoSimpleFoam(
    char* argsAll,
    PyObject* pyOptions)
    : DASolver(argsAll, pyOptions),
      simplePtr_(nullptr),
      pThermoPtr_(nullptr),
      pPtr_(nullptr),
      rhoPtr_(nullptr),
      UPtr_(nullptr),
      phiPtr_(nullptr),
      pressureControlPtr_(nullptr),
      turbulencePtr_(nullptr),
      daTurbulenceModelPtr_(nullptr),
      daFvSourcePtr_(nullptr),
      fvSourcePtr_(nullptr),
      fvSourceEnergyPtr_(nullptr),
      initialMass_(dimensionedScalar("initialMass", dimensionSet(1, 0, 0, 0, 0, 0, 0), 0.0)),
      MRFPtr_(nullptr)
{
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

void DARhoSimpleFoam::initSolver()
{
    /*
    Description:
        Initialize variables for DASolver
    */

    Info << "Initializing fields for DARhoSimpleFoam" << endl;
    Time& runTime = runTimePtr_();
    fvMesh& mesh = meshPtr_();
    argList& args = argsPtr_();
#include "createSimpleControlPython.H"
#include "createFieldsRhoSimple.H"

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

label DARhoSimpleFoam::solvePrimal()
{
    /*
    Description:
        Call the primal solver to get converged state variables
    */

#include "createRefsRhoSimple.H"
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
        rho.storePrevIter();

        // Pressure-velocity SIMPLE corrector
#include "UEqnRhoSimple.H"
#include "EEqnRhoSimple.H"
#include "pEqnRhoSimple.H"

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

} // End namespace Foam

// ************************************************************************* //
