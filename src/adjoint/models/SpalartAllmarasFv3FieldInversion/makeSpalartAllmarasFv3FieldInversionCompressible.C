/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v3

    This file is modified from OpenFOAM's source code
    src/TurbulenceModels/compressible/turbulentFluidThermoModels/turbulentFluidThermoModels.H

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

#include "CompressibleTurbulenceModel.H"
#include "compressibleTransportModel.H"
#include "fluidThermo.H"
#include "addToRunTimeSelectionTable.H"
#include "makeTurbulenceModel.H"

#include "ThermalDiffusivity.H"
#include "EddyDiffusivity.H"

#include "RASModel.H"
#include "LESModel.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
#define createBaseTurbulenceModel(                          \
    Alpha, Rho, baseModel, BaseModel, TDModel, Transport)   \
                                                            \
    namespace Foam                                          \
    {                                                       \
    typedef TDModel<BaseModel<Transport>>                   \
        Transport##BaseModel;                               \
    typedef RASModel<EddyDiffusivity<Transport##BaseModel>> \
        RAS##Transport##BaseModel;                          \
    typedef LESModel<EddyDiffusivity<Transport##BaseModel>> \
        LES##Transport##BaseModel;                          \
    }

createBaseTurbulenceModel(
    geometricOneField,
    volScalarField,
    compressibleTurbulenceModel,
    CompressibleTurbulenceModel,
    ThermalDiffusivity,
    fluidThermo);

#define makeRASModel(Type) \
    makeTemplatedTurbulenceModel(fluidThermoCompressibleTurbulenceModel, RAS, Type)

#define makeLESModel(Type) \
    makeTemplatedTurbulenceModel(fluidThermoCompressibleTurbulenceModel, LES, Type)

#include "SpalartAllmarasFv3FieldInversion.H"
makeRASModel(SpalartAllmarasFv3FieldInversion);

// ************************************************************************* //
