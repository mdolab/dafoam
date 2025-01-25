/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

    Description:
        Child class for regression parameters

\*---------------------------------------------------------------------------*/

#ifndef DAInputRegressionPar_H
#define DAInputRegressionPar_H

#include "DAInput.H"
#include "addToRunTimeSelectionTable.H"
#include "DARegression.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
      Class DAInputRegressionPar Declaration
\*---------------------------------------------------------------------------*/

class DAInputRegressionPar
    : public DAInput
{

protected:
    /// list of patch names for the thermal var
    DARegression& daRegression_;

public:
    TypeName("regressionPar");
    // Constructors

    //- Construct from components
    DAInputRegressionPar(
        const word inputName,
        const word inputType,
        fvMesh& mesh,
        const DAOption& daOption,
        const DAModel& daModel,
        const DAIndex& daIndex);

    //- Destructor
    virtual ~DAInputRegressionPar()
    {
    }

    virtual void run(const scalarList& input);

    virtual label size()
    {
        return daRegression_.nParameters(inputName_);
    }

    virtual label distributed()
    {
        return 0;
    }
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
