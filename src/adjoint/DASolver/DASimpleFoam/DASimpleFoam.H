/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

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

    /// Prandtl number pointer
    autoPtr<dimensionedScalar> PrPtr_;

    /// Turbulence Prandtl pointer
    autoPtr<dimensionedScalar> PrtPtr_;

    /// temperature field pointer
    autoPtr<volScalarField> TPtr_;

    /// thermal diffusivity field pointer
    autoPtr<volScalarField> alphatPtr_;

    /// whether to have the temperature field
    label hasTField_;

#ifdef CODI_ADR
    /// tape positions for consistent fixed-point adjoint
    using Position = typename codi::RealReverse::Tape::Position;
    Position adjResStart_;
    Position adjResEnd_;
    Position gradPStart_;
    Position gradPEnd_;
    Position pEqnFluxStart_;
    Position pEqnFluxEnd_;
    Position divPhiStart_;
    Position divPhiEnd_;
    Position fluxUStart_;
    Position fluxUEnd_;
#endif

    /// fvSolution parameters for fixed-point adjoint
    scalar relaxUEqn_ = 1.0;
    dictionary solverDictU_;
    dictionary solverDictP_;

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
    virtual label solvePrimal();

    /// solve the adjoint equation using the fixed-point iteration method
    virtual label runFPAdj(
        Vec dFdW,
        Vec psi);

    /// Inverse transpose product, MU^(-T)
    void invTranProdUEqn(
        const volVectorField& mySource,
        volVectorField& pseudoU);

    /// Inverse transpose product, Mp^(-T)
    void invTranProdPEqn(
        const volScalarField& mySource,
        volScalarField& pseudoP);

    /// calculate all the flow residuals except for the turbulence one
    void calcLduResiduals(
        volVectorField& URes,
        volScalarField& pRes,
        surfaceScalarField& phiRes);

    /// assign coupled petsc vec to fields or vice versa
    void vec2Fields(
        const word mode,
        Vec cVec,
        volVectorField& UField,
        volScalarField& pField,
        surfaceScalarField& phiField,
        volScalarField& nuTildaField);

    void calcAdjointResidual(
        volVectorField& URes,
        volScalarField& pRes,
        surfaceScalarField& phiRes,
        volScalarField& nuTildaRes,
        Vec dFdW,
        volVectorField& UPsi,
        volScalarField& pPsi,
        surfaceScalarField& phiPsi,
        volScalarField& nuTildaPsi,
        volVectorField& adjURes,
        volScalarField& adjPRes,
        surfaceScalarField& adjPhiRes,
        volScalarField& adjNuTildaRes,
        label& cnt);

    scalar L2norm(const volScalarField& v);

    vector L2norm(const volVectorField& U);

    scalar L2norm(const surfaceScalarField& Phi);

    void swap(List<scalar>& a, List<scalar>& b);
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
