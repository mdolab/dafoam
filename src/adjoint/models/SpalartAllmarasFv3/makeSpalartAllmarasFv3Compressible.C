/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v2

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

#include "SpalartAllmarasFv3.H"
makeRASModel(SpalartAllmarasFv3);

// ************************************************************************* //
