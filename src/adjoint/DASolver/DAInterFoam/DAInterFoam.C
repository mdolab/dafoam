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

#include "DAInterFoam.H"

#include "fvCFD.H"
#include "CMULES.H"
#include "EulerDdtScheme.H"
#include "localEulerDdtScheme.H"
#include "CrankNicolsonDdtScheme.H"
#include "subCycle.H"
#include "immiscibleIncompressibleTwoPhaseMixture.H"
#include "turbulentTransportModel.H"
#include "pimpleControl.H"
#include "fvOptions.H"
#include "fvcSmooth.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(DAInterFoam, 0);
addToRunTimeSelectionTable(DASolver, DAInterFoam, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DAInterFoam::DAInterFoam(
    char* argsAll,
    PyObject* pyOptions)
    : DASolver(argsAll, pyOptions)
{
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

void DAInterFoam::initSolver()
{
    /*
    Description:
        Initialize variables for DASolver
    */
}

label DAInterFoam::solvePrimal()
{
    /*
    Description:
        Call the primal solver to get converged state variables
    */

    Foam::argList& args = argsPtr_();

#include "createTime.H"
#include "createMesh.H"
#include "initContinuityErrs.H"
#include "createControl.H"
#include "createTimeControls.H"
#include "createFieldsInter.H"
#include "createAlphaFluxes.H"
#include "createFvOptions.H"

    turbulence->validate();

#include "readTimeControls.H"
#include "CourantNo.H"
#include "setInitialDeltaT.H"

    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
    Info << "\nStarting time loop\n"
         << endl;

    while (runTime.run())
    {

#include "readTimeControls.H"
#include "CourantNo.H"
#include "alphaCourantNo.H"
#include "setDeltaT.H"

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

            if (pimple.turbCorr())
            {
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
