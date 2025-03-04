/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

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
    : public regIOobject
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

    /// whether the regression model is active
    label active_;

    /// name of the regression models
    wordList modelNames_;

    /// the type of regression model
    HashTable<word> modelType_;

    /// a list of words for the inputs
    HashTable<wordList> inputNames_;

    /// a list of words for the outputs
    HashTable<word> outputName_;

    /// number of neurons hidden layers of the neural network
    HashTable<labelList> hiddenLayerNeurons_;

    /// we can shift each input. we always shift before scaling it.
    HashTable<scalarList> inputShift_;

    /// we can scale each input. we always shift before scaling it.
    HashTable<scalarList> inputScale_;

    /// we can shift the output. we always shift before scaling it.
    HashTable<scalar> outputShift_;

    /// we can scale the output. we always shift before scaling it.
    HashTable<scalar> outputScale_;

    /// the parameters for the regression model
    HashTable<scalarList> parameters_;

    /// neural network activation function
    HashTable<word> activationFunction_;

    /// if the ReLU activation function is used we can prescribe a potentially leaky coefficient
    HashTable<scalar> leakyCoeff_;

    /// the upper bound for the output
    HashTable<scalar> outputUpperBound_;

    /// the lower bound for the output
    HashTable<scalar> outputLowerBound_;

    /// whether to print the input range info this is used to scale the input
    HashTable<label> printInputInfo_;

    /// default output values
    HashTable<scalar> defaultOutputValue_;

    /// number of radial basis function
    HashTable<label> nRBFs_;

    /// a list of scalarField to save the input features
    HashTable<PtrList<volScalarField>> features_;

#if defined(CODI_ADF)
    /// a flatten double feature array for externalTensorFlow model (forward AD)
    double* featuresFlattenArrayDouble_;

    /// output field double array for externalTensorFlow model (forward AD)
    double* outputFieldArrayDouble_;
#else
    /// a flatten feature array for externalTensorFlow model
    scalar* featuresFlattenArray_;

    /// output field array for externalTensorFlow model
    scalar* outputFieldArray_;
#endif
    /// whether to use external model
    label useExternalModel_ = 0;

    /// the array size is chosen based on the regModel that has the largest number of inputs (this is important because we support multiple regModels but we don't want to create multiple featuresFlattenArray_)
    label featuresFlattenArraySize_ = -100;

    /// whether to write the feature fields to the disk
    HashTable<label> writeFeatures_;

public:
    /// Constructors
    DARegression(
        const fvMesh& mesh,
        const DAOption& daOption,
        const DAModel& daModel);

    /// Destructor
    virtual ~DARegression()
    {
        if (useExternalModel_)
        {
#if defined(CODI_ADF)
            delete[] featuresFlattenArrayDouble_;
            delete[] outputFieldArrayDouble_;
#else
            delete[] featuresFlattenArray_;
            delete[] outputFieldArray_;
#endif
        }
    }

    // Members

    /// compute the output based on the latest parameters and inputs
    label compute();

    /// calculate the input flow features
    void calcInputFeatures(word modelName);

    /// get the number of parameters for this regression model
    label nParameters(word modelName);

    /// get a specific parameter value
    scalar getParameter(word modelName, label idxI)
    {
        return parameters_[modelName][idxI];
    }

    /// set a value to a parameter give an index and a value
    void setParameter(word modelName, label idxI, scalar val)
    {
        parameters_[modelName][idxI] = val;
    }

    /// check if the output values are valid otherwise bound or fix them
    label checkOutput(word modelName, volScalarField& outputField);

    /// print the input
    void printInputInfo(const label printToScreen)
    {
        if (!active_ || !printToScreen)
        {
            return;
        }
        forAll(modelNames_, idxI)
        {
            word modelName = modelNames_[idxI];
            if (printInputInfo_[modelName])
            {
                Info << "RegModel input info for " << modelName << endl;
                forAll(inputNames_[modelName], idxI)
                {
                    word name = inputNames_[modelName][idxI];
                    Info << name << " Min: " << gMin(features_[modelName][idxI]) << " Max: " << gMax(features_[modelName][idxI])
                         << " Avg: " << gAverage(features_[modelName][idxI]) << " Std: " << sqrt(gSumSqr(features_[modelName][idxI]) / features_[modelName][idxI].size()) << endl;
                }
            }
        }
    }

    /// write the features to the disk
    void writeFeatures()
    {
        if (!active_)
        {
            return;
        }
        forAll(modelNames_, idxI)
        {
            word modelName = modelNames_[idxI];
            if (writeFeatures_[modelName])
            {
                forAll(features_[modelName], idxI)
                {
                    features_[modelName][idxI].write();
                }
            }
        }
    }

#ifdef CODI_ADR
/*
    /// these two functions are for AD external functions
    static void betaCompute(
        const double* x,
        size_t n,
        double* y,
        size_t m,
        codi::ExternalFunctionUserData* d)
    {
        DAUtility::pyCalcBetaInterface(x, n, y, m, DAUtility::pyCalcBeta);
    }

    static void betaJacVecProd(
        const double* x,
        double* x_b,
        size_t n,
        const double* y,
        const double* y_b,
        size_t m,
        codi::ExternalFunctionUserData* d)
    {
        DAUtility::pyCalcBetaJacVecProdInterface(x, x_b, n, y, y_b, m, DAUtility::pyCalcBetaJacVecProd);
    }
*/
#endif

    /// this is a virtual function for regIOobject
    bool writeData(Ostream& os) const
    {
        return true;
    }
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
