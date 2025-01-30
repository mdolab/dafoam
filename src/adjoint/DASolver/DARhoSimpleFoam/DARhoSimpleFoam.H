/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

    Description:
        Child class for DARhoSimpleFoam
    
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

#ifndef DARhoSimpleFoam_H
#define DARhoSimpleFoam_H

#include "DASolver.H"
#include "addToRunTimeSelectionTable.H"
#include "fluidThermo.H"
#include "turbulentFluidThermoModel.H"
#include "simpleControl.H"
#include "pressureControl.H"
#include "DAFvSource.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
      Class DARhoSimpleFoam Declaration
\*---------------------------------------------------------------------------*/

class DARhoSimpleFoam
    : public DASolver
{

protected:
    /// simple pointer
    autoPtr<simpleControl> simplePtr_;

    /// thermal model
    autoPtr<fluidThermo> pThermoPtr_;

    /// pressure field pointer
    autoPtr<volScalarField> pPtr_;

    /// density field pointer
    autoPtr<volScalarField> rhoPtr_;

    /// velocity field pointer
    autoPtr<volVectorField> UPtr_;

    /// surface flux field pointer
    autoPtr<surfaceScalarField> phiPtr_;

    /// pressure control pointer
    autoPtr<pressureControl> pressureControlPtr_;

    /// turbulence pointer
    autoPtr<compressible::turbulenceModel> turbulencePtr_;

    /// DATurbulenceModel pointer
    autoPtr<DATurbulenceModel> daTurbulenceModelPtr_;

    /// DASource pointer
    autoPtr<DAFvSource> daFvSourcePtr_;

    /// fvSource term
    autoPtr<volVectorField> fvSourcePtr_;

    /// fvSource term for the energy equation
    autoPtr<volScalarField> fvSourceEnergyPtr_;

    /// whether to have fvSource term
    label hasFvSource_ = 0;

    /// initial mass
    dimensionedScalar initialMass_;

    /// MRF pointer
    autoPtr<IOMRFZoneListDF> MRFPtr_;

    /// continuity error
    scalar cumulativeContErr_ = 0.0;

public:
    TypeName("DARhoSimpleFoam");
    // Constructors

    //- Construct from components
    DARhoSimpleFoam(
        char* argsAll,
        PyObject* pyOptions);

    //- Destructor
    virtual ~DARhoSimpleFoam()
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
