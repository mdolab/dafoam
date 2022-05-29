/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v3

    Description:
        Child class for DASimpleFoam
    
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

#ifndef DASimpleFoam_H
#define DASimpleFoam_H

#include "DASolver.H"
#include "addToRunTimeSelectionTable.H"
#include "singlePhaseTransportModel.H"
#include "turbulentTransportModel.H"
#include "simpleControl.H"
#include "DARegDbSinglePhaseTransportModel.H"
#include "DARegDbTurbulenceModelIncompressible.H"
#include "DAFvSource.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
      Class DASimpleFoam Declaration
\*---------------------------------------------------------------------------*/

class DASimpleFoam
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

public:
    TypeName("DASimpleFoam");
    // Constructors

    //- Construct from components
    DASimpleFoam(
        char* argsAll,
        PyObject* pyOptions);

    //- Destructor
    virtual ~DASimpleFoam()
    {
    }

    /// initialize fields and variables
    virtual void initSolver();

    /// solve the primal equations
    virtual label solvePrimal(
        const Vec xvVec,
        Vec wVec);

    /// solve the adjoint equation using the fixed-point iteration method
    virtual label runFPAdj(
        Vec dFdW,
        Vec psi);

    /// Inverse transpose product, MU^(-T)
    void invTranProdUEqn(
        const List<vector>& mySource,
        volVectorField& pseudo_U);

    /// Inverse transpose product, Mp^(-T)
    void invTranProdPEqn(
        const List<scalar>& mySource,
        volScalarField& pseudo_p);

    /// calculate all the flow residuals except for the turbulence one
    void calcLduResiduals(
        volVectorField& URes,
        volScalarField& pRes,
        surfaceScalarField& phiRes);

    void calcAdjointResidual(
        volVectorField& URes,
        volScalarField& pRes,
        surfaceScalarField& phiRes,
        volScalarField& nuTildaRes,
        List<vector>& dfdU,
        List<scalar>& dfdp,
        surfaceScalarField& dfdphi,
        List<scalar>& dfdnuTilda,
        List<vector>& UPsi,
        List<scalar>& pPsi,
        surfaceScalarField& phiPsi,
        List<scalar>& nuTildaPsi,
        List<vector>& adjURes,
        List<scalar>& adjpRes,
        surfaceScalarField& adjphiRes,
        List<scalar>& adjnuTildaRes,
        label& cnt);
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
