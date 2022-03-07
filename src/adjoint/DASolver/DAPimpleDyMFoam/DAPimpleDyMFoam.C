/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v3

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

#include "DAPimpleDyMFoam.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(DAPimpleDyMFoam, 0);
addToRunTimeSelectionTable(DASolver, DAPimpleDyMFoam, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DAPimpleDyMFoam::DAPimpleDyMFoam(
    char* argsAll,
    PyObject* pyOptions)
    : DASolver(argsAll, pyOptions),
      daMotionPtr_(nullptr)
{
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

void DAPimpleDyMFoam::initSolver()
{
    /*
    Description:
        Initialize variables for DASolver
    */
    daOptionPtr_.reset(new DAOption(meshPtr_(), pyOptions_));
}

label DAPimpleDyMFoam::solvePrimal(
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
    Foam::argList& args = argsPtr_();
#include "createTime.H"
#include "createDynamicFvMesh.H"
#include "initContinuityErrs.H"
#include "createDyMControls.H"
#include "createFieldsPimpleDyM.H"
#include "createUfIfPresent.H"
#include "CourantNo.H"
#include "setInitialDeltaT.H"

    daMotionPtr_.reset(DAMotion::New(mesh, daOptionPtr_()));

    turbulence->validate();

    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

    Info << "\nStarting time loop\n"
         << endl;

    while (runTime.run())
    {

#include "readDyMControls.H"
#include "CourantNo.H"
#include "setDeltaT.H"

        ++runTime;

        daMotionPtr_->correct();

        Info << "Time = " << runTime.timeName() << nl << endl;

        // --- Pressure-velocity PIMPLE corrector loop
        while (pimple.loop())
        {
            if (pimple.firstIter() || moveMeshOuterCorrectors)
            {
                mesh.update();

                if (mesh.changing())
                {
                    MRF.update();

                    if (correctPhi)
                    {
                        // Calculate absolute flux
                        // from the mapped surface velocity
                        phi = mesh.Sf() & Uf();

#include "correctPhiPimpleDyM.H"

                        // Make the flux relative to the mesh motion
                        fvc::makeRelative(phi, U);
                    }

                    if (checkMeshCourantNo)
                    {
#include "meshCourantNo.H"
                    }
                }
            }

#include "UEqnPimpleDyM.H"

            // --- Pressure corrector loop
            while (pimple.correct())
            {
#include "pEqnPimpleDyM.H"
            }

            if (pimple.turbCorr())
            {
                laminarTransport.correct();
                turbulence->correct();
            }
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
