/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

    Description:
        Child class for DAPimpleDyMFoam
    
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

#ifndef DAPimpleDyMFoam_H
#define DAPimpleDyMFoam_H

#include "DASolver.H"
#include "addToRunTimeSelectionTable.H"
#include "singlePhaseTransportModel.H"
#include "turbulentTransportModel.H"
#include "pimpleControlDF.H"
#include "pimpleControl.H"
//#include "dynamicFvMesh.H"
#include "CorrectPhi.H"

void myCorrectUf
(
    autoPtr<surfaceVectorField>& Uf,
    const volVectorField& U,
    const surfaceScalarField& phi
)
{
    const fvMesh& mesh = U.mesh();

    if (mesh.moving())
    {
        Uf() = fvc::interpolate(U);
        surfaceVectorField n(mesh.Sf()/mesh.magSf());
        Uf() += n*(phi/mesh.magSf() - (n & Uf()));
    }
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
      Class DAPimpleDyMFoam Declaration
\*---------------------------------------------------------------------------*/

class DAPimpleDyMFoam
    : public DASolver
{

protected:
public:
    TypeName("DAPimpleDyMFoam");
    // Constructors

    //- Construct from components
    DAPimpleDyMFoam(
        char* argsAll,
        PyObject* pyOptions);

    //- Destructor
    virtual ~DAPimpleDyMFoam()
    {
    }

    /// initialize fields and variables
    virtual void initSolver();

    /// solve the primal equations
    virtual label solvePrimal();
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
