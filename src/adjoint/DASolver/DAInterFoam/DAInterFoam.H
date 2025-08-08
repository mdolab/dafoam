/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

    Description:
        Child class for DAInterFoam
    
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

#ifndef DAInterFoam_H
#define DAInterFoam_H

#include "DASolver.H"
#include "addToRunTimeSelectionTable.H"
#include "fvCFD.H"
#include "CMULESDF.H"
#include "EulerDdtScheme.H"
#include "localEulerDdtScheme.H"
#include "CrankNicolsonDdtScheme.H"
#include "subCycle.H"
#include "immiscibleIncompressibleTwoPhaseMixture.H"
#include "turbulentTransportModel.H"
#include "pimpleControl.H"
#include "pimpleControlDF.H"
#include "fvcSmooth.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
      Class DAInterFoam Declaration
\*---------------------------------------------------------------------------*/

class DAInterFoam
    : public DASolver
{

protected:
    /// pimple pointer
    autoPtr<pimpleControlDF> pimplePtr_;

    /// pressure - rho*g*h field pointer
    autoPtr<volScalarField> p_rghPtr_;

    /// pressure field pointer
    autoPtr<volScalarField> pPtr_;

    /// velocity field pointer
    autoPtr<volVectorField> UPtr_;

    /// density field pointer
    autoPtr<volScalarField> rhoPtr_;

    /// surface flux field pointer
    autoPtr<surfaceScalarField> phiPtr_;

    /// density times surface flux field pointer
    autoPtr<surfaceScalarField> rhoPhiPtr_;

    /// g*h field pointer
    autoPtr<volScalarField> ghPtr_;

    /// g*h face field pointer
    autoPtr<surfaceScalarField> ghfPtr_;

    /// MULES compressed flux pointer
    autoPtr<surfaceScalarField> alphaPhiUnPtr_;

    /// MULES flux from previous time-step pointer
    autoPtr<surfaceScalarField> alphaPhi10Ptr_;

    /// laminar transport properties pointer
    autoPtr<immiscibleIncompressibleTwoPhaseMixture> mixturePtr_;

    /// turbulence pointer
    autoPtr<incompressible::turbulenceModel> turbulencePtr_;

    /// DATurbulenceModel pointer
    autoPtr<DATurbulenceModel> daTurbulenceModelPtr_;

    /// continuity error
    scalar cumulativeContErr_ = 0.0;

    /// pressure referefence cell id
    label pRefCell_ = 0;

    /// pressure reference value
    scalar pRefValue_ = 0.0;

    /// the primal IO interval
    label IOInterval_ = 1;

    /// whether to write mesh for the reduceIO
    label reduceIOWriteMesh_ = 0;

    /// whether is a restart run
    bool alphaRestart_ = false;

public:
    TypeName("DAInterFoam");
    // Constructors

    //- Construct from components
    DAInterFoam(
        char* argsAll,
        PyObject* pyOptions);

    //- Destructor
    virtual ~DAInterFoam()
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
