/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v3

    This file is modified from OpenFOAM's source code
    src/TurbulenceModels/incompressible/turbulentTransportModels/turbulentTransportModels.H

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

#include "IncompressibleTurbulenceModel.H"
#include "transportModel.H"
#include "addToRunTimeSelectionTable.H"
#include "makeTurbulenceModel.H"

#include "RASModel.H"
#include "LESModel.H"
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
#define createBaseTurbulenceModel(Alpha, Rho, baseModel, BaseModel, Transport) \
                                                                               \
    namespace Foam                                                             \
    {                                                                          \
    typedef BaseModel<Transport> Transport##BaseModel;                         \
    typedef RASModel<Transport##BaseModel> RAS##Transport##BaseModel;          \
    typedef LESModel<Transport##BaseModel> LES##Transport##BaseModel;          \
    }
createBaseTurbulenceModel(
    geometricOneField,
    geometricOneField,
    incompressibleTurbulenceModel,
    IncompressibleTurbulenceModel,
    transportModel);
#define makeRASModel(Type) \
    makeTemplatedTurbulenceModel(transportModelIncompressibleTurbulenceModel, RAS, Type)
#define makeLESModel(Type) \
    makeTemplatedTurbulenceModel(transportModelIncompressibleTurbulenceModel, LES, Type)

#include "SpalartAllmarasFv3FieldInversion.H"
makeRASModel(SpalartAllmarasFv3FieldInversion);

// ************************************************************************* //
