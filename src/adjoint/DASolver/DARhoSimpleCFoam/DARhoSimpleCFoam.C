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

#include "DARhoSimpleCFoam.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(DARhoSimpleCFoam, 0);
addToRunTimeSelectionTable(DASolver, DARhoSimpleCFoam, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DARhoSimpleCFoam::DARhoSimpleCFoam(
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
      initialMass_(dimensionedScalar("initialMass", dimensionSet(1, 0, 0, 0, 0, 0, 0), 0.0))
{
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

void DARhoSimpleCFoam::initSolver()
{
    /*
    Description:
        Initialize variables for DASolver
    */

    Info << "Initializing fields for DARhoSimpleCFoam" << endl;
    Time& runTime = runTimePtr_();
    fvMesh& mesh = meshPtr_();
    argList& args = argsPtr_();
#include "createSimpleControlPython.H"
#include "createFieldsRhoSimpleC.H"

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
}

label DARhoSimpleCFoam::solvePrimal()
{
    /*
    Description:
        Call the primal solver to get converged state variables
    */

#include "createRefsRhoSimpleC.H"

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
#include "UEqnRhoSimpleC.H"
#include "EEqnRhoSimpleC.H"
#include "pEqnRhoSimpleC.H"

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

    Info << "End\n"
         << endl;

    return this->checkPrimalFailure();
}

} // End namespace Foam

// ************************************************************************* //
