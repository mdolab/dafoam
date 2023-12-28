/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v3

    Description:
        Regression model

\*---------------------------------------------------------------------------*/

#ifndef DARegression_H
#define DARegression_H

#include "fvOptions.H"
#include "surfaceFields.H"
#include "DAOption.H"
#include "DAUtility.H"
#include "DAModel.H"
#include "globalIndex.H"
#include "DAMacroFunctions.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                       Class DARegression Declaration
\*---------------------------------------------------------------------------*/

class DARegression
{

private:
    /// Disallow default bitwise copy construct
    DARegression(const DARegression&);

    /// Disallow default bitwise assignment
    void operator=(const DARegression&);

protected:
    /// Foam::fvMesh object
    const fvMesh& mesh_;

    /// Foam::DAOption object
    const DAOption& daOption_;

    /// DAModel object
    const DAModel& daModel_;

    /// the type of regression model
    word modelType_;

    /// a list of words for the inputs
    wordList inputNames_;

    /// a list of words for the outputs
    word outputName_;

    /// number of neurons hidden layers of the neural network
    labelList hiddenLayerNeurons_;

    /// we can shift the output. we always shift before scaling it.
    scalar outputShift_;

    /// we can scale the output. we always shift before scaling it.
    scalar outputScale_;

    /// the parameters for the regression model
    scalarList parameters_;

public:
    /// Constructors
    DARegression(
        const fvMesh& mesh,
        const DAOption& daOption,
        const DAModel& daModel);

    /// Destructor
    virtual ~DARegression()
    {
    }

    // Members

    void compute();

    label nParameters();

    scalar getParameter(label idxI)
    {
        return parameters_[idxI];
    }

    void setParameter(label idxI, scalar val)
    {
        parameters_[idxI] = val;
    }

};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
