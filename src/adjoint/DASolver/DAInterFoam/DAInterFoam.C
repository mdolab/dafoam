/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

    This class is modified from OpenFOAM's source code
    applications/solvers/multiphase/interFoam

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

#include "DAInterFoam.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(DAInterFoam, 0);
addToRunTimeSelectionTable(DASolver, DAInterFoam, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DAInterFoam::DAInterFoam(
    char* argsAll,
    PyObject* pyOptions)
    : DASolver(argsAll, pyOptions),
      pimplePtr_(nullptr),
      p_rghPtr_(nullptr),
      pPtr_(nullptr),
      UPtr_(nullptr),
      rhoPtr_(nullptr),
      phiPtr_(nullptr),
      rhoPhiPtr_(nullptr),
      ghPtr_(nullptr),
      ghfPtr_(nullptr),
      alphaPhiUnPtr_(nullptr),
      alphaPhi10Ptr_(nullptr),
      mixturePtr_(nullptr),
      turbulencePtr_(nullptr),
      daTurbulenceModelPtr_(nullptr)
{
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

void DAInterFoam::initSolver()
{
    /*
    Description:
        Initialize variables for DASolver
    */

    Info << "Initializing fields for DAInterFoam" << endl;
    Time& runTime = runTimePtr_();
    fvMesh& mesh = meshPtr_();

#include "createPimpleControlPython.H"
#include "createFieldsInter.H"

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

label DAInterFoam::solvePrimal()
{
    /*
    Description:
        Call the primal solver to get converged state variables
    */

#include "createRefsInter.H"

    // call correctNut, this is equivalent to turbulence->validate();
    daTurbulenceModelPtr_->updateIntermediateVariables();

    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
    Info << "\nStarting time loop\n"
         << endl;

    while (runTime.run())
    {

#include "CourantNo.H"
#include "alphaCourantNo.H"

        runTime++;

        Info << "Time = " << runTime.timeName() << nl << endl;

        // --- Pressure-velocity PIMPLE corrector loop
        while (pimple.loop())
        {

#include "alphaControls.H"
#include "alphaEqnSubCycle.H"

            mixture.correct();

            if (pimple.frozenFlow())
            {
                continue;
            }

#include "UEqnInter.H"

            // --- Pressure corrector loop
            while (pimple.correct())
            {
#include "pEqnInter.H"
            }

            turbulencePtr_->correct();
        }

        runTime.write();

        runTime.printExecutionTime(Info);
    }

    Info << "End\n"
         << endl;
    return 0;
}

} // End namespace Foam

// ************************************************************************* //
