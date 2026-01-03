/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v5

    Description:
        Child class for DATopoChtFoamFoam

        Topology optimization considering conjugate heat transfer
    
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

#ifndef DATopoChtFoam_H
#define DATopoChtFoam_H

#include "DASolver.H"
#include "addToRunTimeSelectionTable.H"
#include "singlePhaseTransportModel.H"
#include "turbulentTransportModel.H"
#include "simpleControl.H"
#include "DAFvSource.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
      Class DATopoChtFoam Declaration
\*---------------------------------------------------------------------------*/

class DATopoChtFoam
    : public DASolver
{

protected:
    /// simple pointer
    autoPtr<simpleControl> simplePtr_;

    /// pressure field pointer
    autoPtr<volScalarField> pPtr_;

    /// velocity field pointer
    autoPtr<volVectorField> UPtr_;

    /// surface flux field pointer
    autoPtr<surfaceScalarField> phiPtr_;

    /// alpha porosity
    autoPtr<volScalarField> alphaPorosityPtr_;

    /// laminar transport properties pointer
    autoPtr<singlePhaseTransportModel> laminarTransportPtr_;

    /// turbulence pointer
    autoPtr<incompressible::turbulenceModel> turbulencePtr_;

    /// DATurbulenceModel pointer
    autoPtr<DATurbulenceModel> daTurbulenceModelPtr_;

    /// DASource pointer
    autoPtr<DAFvSource> daFvSourcePtr_;

    /// fvSource term
    autoPtr<volVectorField> fvSourcePtr_;

    /// MRF pointer
    autoPtr<IOMRFZoneListDF> MRFPtr_;

    /// whether to have fvSource term
    label hasFvSource_ = 0;

    /// continuity error
    scalar cumulativeContErr_ = 0.0;

    /// pressure referefence cell id
    label pRefCell_ = 0;

    /// pressure reference value
    scalar pRefValue_ = 0.0;

    /// Prandtl number pointer
    autoPtr<dimensionedScalar> PrPtr_;

    /// Turbulence Prandtl pointer
    autoPtr<dimensionedScalar> PrtPtr_;

    /// temperature field pointer
    autoPtr<volScalarField> TPtr_;

    /// total diffusivity field pointer
    autoPtr<volScalarField> diffusivityPtr_;

    /// cell-wise design variable topology optimization [0:1]
    autoPtr<volScalarField> etaPtr_;

    /// these are properties for the fluid and solid
    scalar alphaPorosityFluid_;
    scalar alphaPorositySolid_;
    autoPtr<volScalarField> diffusivityFluidPtr_;
    scalar diffusivitySolid_;

    /// penalty coefficient
    scalar pCoeff_;

public:
    TypeName("DATopoChtFoam");
    // Constructors

    //- Construct from components
    DATopoChtFoam(
        char* argsAll,
        PyObject* pyOptions);

    //- Destructor
    virtual ~DATopoChtFoam()
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
