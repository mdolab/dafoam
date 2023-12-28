/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v3

\*---------------------------------------------------------------------------*/

#include "DARegression.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DARegression::DARegression(
    const fvMesh& mesh,
    const DAOption& daOption,
    const DAModel& daModel)
    : mesh_(mesh),
      daOption_(daOption),
      daModel_(daModel)
{
    dictionary regSubDict = daOption.getAllOptions().subDict("regressionModel");
    regSubDict.readEntry<word>("modelType", modelType_);
    regSubDict.readEntry<wordList>("inputNames", inputNames_);
    regSubDict.readEntry<word>("outputName", outputName_);

    regSubDict.readEntry<scalar>("outputShift", outputShift_);

    regSubDict.readEntry<scalar>("outputScale", outputScale_);

    active_ = regSubDict.getLabel("active");

    if (modelType_ == "neuralNetwork")
    {
        regSubDict.readEntry<labelList>("hiddenLayerNeurons", hiddenLayerNeurons_);
    }

    // initialize parameters and give it large values
    if (active_)
    {
        label nParameters = this->nParameters();
        parameters_.setSize(nParameters);
        forAll(parameters_, idxI)
        {
            parameters_[idxI] = 1e16;
        }
    }
}

void DARegression::compute()
{
    /*
    Description:
        Calculate the prescribed output field based on the prescribed input fields using a regression model.
        We support only neural network at this moment

    Input:
        parameters_:
            the parameters for the regression. For the neural network model, these parameters
            are the weights and biases. We order them in an 1D array, starting from the first input's weight and biases.

        daOption Dict: regressionModel
            inputNames: a list of volScalarFields prescribed by inputFields
            hiddenLayerNeurons: number of neurons for each hidden layer. 
            example: {5, 3, 4} means three hidden layers with 5, 3, and 4 neurons.

    Output:
        a volScalarField prescribed by outputName
    */

    if (!active_)
    {
        return;
    }

    volScalarField& outputField = const_cast<volScalarField&>(mesh_.thisDb().lookupObject<volScalarField>(outputName_));

    if (modelType_ == "neuralNetwork")
    {
        label nHiddenLayers = hiddenLayerNeurons_.size();

        List<List<scalar>> inputFields;
        inputFields.setSize(inputNames_.size());

        forAll(inputNames_, idxI)
        {
            inputFields[idxI].setSize(mesh_.nCells());
            word inputName = inputNames_[idxI];
            if (inputName == "SoQ")
            {
                // Shear / vorticity
                const volVectorField& U = mesh_.thisDb().lookupObject<volVectorField>("U");
                const tmp<volTensorField> tgradU(fvc::grad(U));
                const volTensorField& gradU = tgradU();
                volScalarField magOmega = mag(skew(gradU));
                volScalarField magS = mag(symm(gradU));
                forAll(inputFields[idxI], cellI)
                {
                    inputFields[idxI][cellI] = magS[cellI] / magOmega[cellI];
                }
            }
            else if (inputName == "PoD")
            {
                // production / destruction
                daModel_.getTurbProdOverDestruct(inputFields[idxI]);
            }
            else if (inputName == "chiSA")
            {
#ifndef SolidDASolver
                // the chi() function from SA
                const volScalarField& nuTilda = mesh_.thisDb().lookupObject<volScalarField>("nuTilda");
                volScalarField nu = daModel_.getDATurbulenceModel().nu();
                forAll(inputFields[idxI], cellI)
                {
                    inputFields[idxI][cellI] = nuTilda[cellI] / nu[cellI];
                }
#endif
            }
            else if (inputName == "pGradStream")
            {
                // pressure gradient along stream
                const volScalarField& p = mesh_.thisDb().lookupObject<volScalarField>("p");
                const volVectorField& U = mesh_.thisDb().lookupObject<volVectorField>("U");
                volVectorField pGrad("gradP", fvc::grad(p));
                volScalarField pG_denominator(mag(U) * mag(pGrad) + mag(U & pGrad));
                volScalarField pGradAlongStream = (U & pGrad) / Foam::max(pG_denominator, dimensionedScalar("minpG", dimensionSet(0, 2, -3, 0, 0, 0, 0), SMALL));
                forAll(inputFields[idxI], cellI)
                {
                    inputFields[idxI][cellI] = pGradAlongStream[cellI];
                }
            }
            else
            {
                FatalErrorIn("") << "inputName: " << inputName << " not supported. Options are: SoQ, PoD, chiSA, pGradStream" << abort(FatalError);
            }
        }

        List<List<scalar>> layerVals;
        layerVals.setSize(nHiddenLayers);
        for (label layerI = 0; layerI < nHiddenLayers; layerI++)
        {
            label nNeurons = hiddenLayerNeurons_[layerI];
            layerVals[layerI].setSize(nNeurons);
        }

        forAll(mesh_.cells(), cellI)
        {
            label counterI = 0;

            for (label layerI = 0; layerI < nHiddenLayers; layerI++)
            {
                label nNeurons = hiddenLayerNeurons_[layerI];
                forAll(layerVals[layerI], neuronI)
                {
                    layerVals[layerI][neuronI] = 0.0;
                }
                for (label neuronI = 0; neuronI < nNeurons; neuronI++)
                {
                    if (layerI == 0)
                    {
                        forAll(inputNames_, neuronJ)
                        {
                            // weighted sum
                            layerVals[layerI][neuronI] += inputFields[neuronJ][cellI] * parameters_[counterI];
                            counterI++;
                        }
                    }
                    else
                    {
                        forAll(layerVals[layerI - 1], neuronJ)
                        {
                            // weighted sum
                            layerVals[layerI][neuronI] += layerVals[layerI - 1][neuronJ] * parameters_[counterI];
                            counterI++;
                        }
                    }
                    // bias
                    layerVals[layerI][neuronI] += parameters_[counterI];
                    counterI++;
                    // activation function
                    layerVals[layerI][neuronI] = 1 / (1 + exp(-layerVals[layerI][neuronI]));
                }
            }
            // final output layer, we have only one output
            scalar outputVal = 0.0;
            forAll(layerVals[nHiddenLayers - 1], neuronJ)
            {
                // weighted sum
                outputVal += layerVals[nHiddenLayers - 1][neuronJ] * parameters_[counterI];
                counterI++;
            }
            // bias
            outputVal += parameters_[counterI];

            outputField[cellI] = outputScale_ * (outputVal + outputShift_);
        }

        outputField.correctBoundaryConditions();
    }
    else
    {
        FatalErrorIn("") << "modelType_: " << modelType_ << " not supported. Options are: neuralNetwork" << abort(FatalError);
    }
}

label DARegression::nParameters()
{
    /*
    Description:
        get the number of parameters
    */

    if (!active_)
    {
        FatalErrorIn("") << "nParameters() is called but the regression model is not active!" << abort(FatalError);
    }

    if (modelType_ == "neuralNetwork")
    {
        label nHiddenLayers = hiddenLayerNeurons_.size();
        label nInputs = inputNames_.size();

        // add weights
        // input
        label nParameters = nInputs * hiddenLayerNeurons_[0];
        // hidden layers
        for (label layerI = 1; layerI < hiddenLayerNeurons_.size(); layerI++)
        {
            nParameters += hiddenLayerNeurons_[layerI] * hiddenLayerNeurons_[layerI - 1];
        }
        // output
        nParameters += hiddenLayerNeurons_[nHiddenLayers - 1] * 1;

        // add biases
        // add hidden layers
        for (label layerI = 0; layerI < hiddenLayerNeurons_.size(); layerI++)
        {
            nParameters += hiddenLayerNeurons_[layerI];
        }
        // add output layer
        nParameters += 1;

        return nParameters;
    }
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
