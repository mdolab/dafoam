/*---------------------------------------------------------------------------*\

    DAFoam : Discrete Adjoint with OpenFOAM

    Description:
        Power (viscous) dissipation objective function
    Usage:
        "function": {
        "viscousDissipation": {
            "type": "DAViscousDissipation",
            "scale": 1.0,
            "source": "allCells"
        }

\*---------------------------------------------------------------------------*/

#ifndef DAFunctionViscousDissipation_H
#define DAFunctionViscousDissipation_H

#include "DAFunction.H"
#include "addToRunTimeSelectionTable.H"

namespace Foam
{

class DAFunctionViscousDissipation
    : public DAFunction
{

protected:
    /// Turbulence model
    const DATurbulenceModel& daTurb_;

public:
    TypeName("DAViscousDissipation");

    DAFunctionViscousDissipation(
        const fvMesh& mesh,
        const DAOption& daOption,
        const DAModel& daModel,
        const DAIndex& daIndex,
        const word functionName);

    virtual ~DAFunctionViscousDissipation()
    {
    }

    virtual scalar calcFunction();
};

}

#endif
