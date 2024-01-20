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

    /// we can shift each input. we always shift before scaling it.
    scalarList inputShift_;

    /// we can scale each input. we always shift before scaling it.
    scalarList inputScale_;

    /// we can shift the output. we always shift before scaling it.
    scalar outputShift_;

    /// we can scale the output. we always shift before scaling it.
    scalar outputScale_;

    /// the parameters for the regression model
    scalarList parameters_;

    /// whether the regression model is active
    label active_;

    /// neural network activation function
    word activationFunction_;

    /// the upper bound for the output
    scalar outputUpperBound_;

    /// the lower bound for the output
    scalar outputLowerBound_;

    /// whether to print the input range info this is used to scale the input
    label printInputRange_;

    /// default output values
    scalar defaultOutputValue_;

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

    /// compute the output based on the latest parameters and inputs
    label compute();

    /// get the number of parameters for this regression model
    label nParameters();

    /// get a specific parameter value
    scalar getParameter(label idxI)
    {
        return parameters_[idxI];
    }

    /// set a value to a parameter give an index and a value
    void setParameter(label idxI, scalar val)
    {
        parameters_[idxI] = val;
    }

    /// check if the regression model is initialized properly
    void validate()
    {
        if (!active_)
        {
            return;
        }
        else
        {
            scalar testVal = this->getParameter(0);
            if (fabs(testVal) > 1e15)
            {
                FatalErrorIn("") << "regressionModel is active but the parameter values are not set!" << abort(FatalError);
            }
        }
    }

    /// check if the output values are valid otherwise bound or fix them
    label checkOutput(volScalarField& outputField);
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
