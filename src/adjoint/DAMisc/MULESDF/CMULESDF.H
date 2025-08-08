/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

    Description:
        A modified version of CMULES from
        src/finiteVolume/fvMatrices/solvers/MULES
        
License
    This file is part of OpenFOAM.

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

Global
    MULES

Description
    CMULES: Multidimensional universal limiter for
    explicit corrected implicit solution.

    Solve a convective-only transport equation using an explicit universal
    multi-dimensional limiter to correct an implicit conservative bounded
    obtained using rigorously bounded schemes such as Euler-implicit in time
    upwind in space.

    Parameters are the variable to solve, the normal convective flux and the
    actual explicit flux of the variable which is also used to return limited
    flux used in the bounded-solution.

SourceFiles
    CMULES.C
    CMULESTemplates.C

\*---------------------------------------------------------------------------*/

#ifndef CMULESDF_H
#define CMULESDF_H

//#include "MULESDF.H"
#include "EulerDdtScheme.H"
#include "localEulerDdtScheme.H"
#include "gaussConvectionScheme.H"
#include "volFieldsFwd.H"
#include "surfaceFieldsFwd.H"
#include "primitiveFieldsFwd.H"
#include "geometricOneField.H"
#include "zero.H"
#include "zeroField.H"
#include "UPtrList.H"
#include "upwind.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace MULESDF
{

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

template<class RdeltaTType, class RhoType, class SpType, class SuType>
void correct(
    const RdeltaTType& rDeltaT,
    const RhoType& rho,
    volScalarField& psi,
    const surfaceScalarField& phi,
    const surfaceScalarField& phiCorr,
    const SpType& Sp,
    const SuType& Su);

template<class RhoType, class SpType, class SuType>
void correct(
    const RhoType& rho,
    volScalarField& psi,
    const surfaceScalarField& phi,
    surfaceScalarField& phiCorr,
    const SpType& Sp,
    const SuType& Su,
    const scalar psiMax,
    const scalar psiMin);

void correct(
    volScalarField& psi,
    const surfaceScalarField& phi,
    surfaceScalarField& phiCorr,
    const scalar psiMax,
    const scalar psiMin);

template<class RdeltaTType, class RhoType, class SpType, class SuType>
void limiterCorr(
    scalarField& allLambda,
    const RdeltaTType& rDeltaT,
    const RhoType& rho,
    const volScalarField& psi,
    const surfaceScalarField& phi,
    const surfaceScalarField& phiCorr,
    const SpType& Sp,
    const SuType& Su,
    const scalar psiMax,
    const scalar psiMin);

template<class RdeltaTType, class RhoType, class SpType, class SuType>
void limitCorr(
    const RdeltaTType& rDeltaT,
    const RhoType& rho,
    const volScalarField& psi,
    const surfaceScalarField& phi,
    surfaceScalarField& phiCorr,
    const SpType& Sp,
    const SuType& Su,
    const scalar psiMax,
    const scalar psiMin);

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace MULESDF
} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#ifdef NoRepository
#include "CMULESDFTemplates.C"
#endif

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
