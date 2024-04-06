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

    regSubDict.readEntry<scalarList>("inputShift", inputShift_);

    regSubDict.readEntry<scalarList>("inputScale", inputScale_);

    regSubDict.readEntry<scalar>("outputShift", outputShift_);

    regSubDict.readEntry<scalar>("outputScale", outputScale_);

    regSubDict.readEntry<scalar>("outputUpperBound", outputUpperBound_);

    regSubDict.readEntry<scalar>("outputLowerBound", outputLowerBound_);

    regSubDict.readEntry<label>("printInputInfo", printInputInfo_);

    regSubDict.readEntry<scalar>("defaultOutputValue", defaultOutputValue_);

    active_ = regSubDict.getLabel("active");

    if (modelType_ == "neuralNetwork")
    {
        regSubDict.readEntry<labelList>("hiddenLayerNeurons", hiddenLayerNeurons_);
        regSubDict.readEntry<word>("activationFunction", activationFunction_);
        if (activationFunction_ == "ReLU")
        {
            leakyCoeff_ = regSubDict.lookupOrDefault<scalar>("leakyCoeff", 0.0);
        }
    }
    else if (modelType_ == "radialBasisFunction")
    {
        nRBFs_ = regSubDict.getLabel("nRBFs");
    }
    else
    {
        FatalErrorIn("") << "modelType_: " << modelType_ << " not supported. Options are: neuralNetwork and radialBasisFunction" << abort(FatalError);
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

void DARegression::calcInput(List<List<scalar>>& inputFields)
{
    /*
    Description:
        Calculate the input features
    */

    forAll(inputNames_, idxI)
    {
        inputFields[idxI].setSize(mesh_.nCells());
        word inputName = inputNames_[idxI];
        if (inputName == "VoS")
        {
            // vorticity / strain
            const volVectorField& U = mesh_.thisDb().lookupObject<volVectorField>("U");
            const tmp<volTensorField> tgradU(fvc::grad(U));
            const volTensorField& gradU = tgradU();
            volScalarField magOmega = mag(skew(gradU));
            volScalarField magS = mag(symm(gradU));
            forAll(inputFields[idxI], cellI)
            {
                inputFields[idxI][cellI] = (magOmega[cellI] / (magS[cellI] + 1e-16) + inputShift_[idxI]) * inputScale_[idxI];
            }
        }
        else if (inputName == "PoD")
        {
            // production / destruction
            daModel_.getTurbProdOverDestruct(inputFields[idxI]);
            forAll(inputFields[idxI], cellI)
            {
                inputFields[idxI][cellI] = (inputFields[idxI][cellI] + inputShift_[idxI]) * inputScale_[idxI];
            }
        }
        else if (inputName == "chiSA")
        {
#ifndef SolidDASolver
            // the chi() function from SA
            const volScalarField& nuTilda = mesh_.thisDb().lookupObject<volScalarField>("nuTilda");
            volScalarField nu = daModel_.getDATurbulenceModel().nu();
            forAll(inputFields[idxI], cellI)
            {
                inputFields[idxI][cellI] = (nuTilda[cellI] / nu[cellI] + inputShift_[idxI]) * inputScale_[idxI];
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
            forAll(pG_denominator, cellI)
            {
                pG_denominator[cellI] += 1e-16;
            }
            volScalarField pGradAlongStream = (U & pGrad) / pG_denominator;
            forAll(inputFields[idxI], cellI)
            {
                inputFields[idxI][cellI] = (pGradAlongStream[cellI] + inputShift_[idxI]) * inputScale_[idxI];
            }
        }
        else if (inputName == "PSoSS")
        {
            // pressure normal stress over shear stress
            const volScalarField& p = mesh_.thisDb().lookupObject<volScalarField>("p");
            const volVectorField& U = mesh_.thisDb().lookupObject<volVectorField>("U");
            const tmp<volTensorField> tgradU(fvc::grad(U));
            const volTensorField& gradU = tgradU();
            volVectorField pGrad("gradP", fvc::grad(p));
            vector diagUGrad = vector::zero;
            scalar val = 0;
            forAll(inputFields[idxI], cellI)
            {
                diagUGrad[0] = gradU[cellI].xx();
                diagUGrad[1] = gradU[cellI].yy();
                diagUGrad[2] = gradU[cellI].zz();
                val = mag(pGrad[cellI]) / (mag(pGrad[cellI]) + mag(3.0 * cmptAv(U[cellI] & diagUGrad)) + 1e-16);
                inputFields[idxI][cellI] = (val + inputShift_[idxI]) * inputScale_[idxI];
            }
        }
        else if (inputName == "SCurv")
        {
            // streamline curvature
            const volVectorField& U = mesh_.thisDb().lookupObject<volVectorField>("U");
            const tmp<volTensorField> tgradU(fvc::grad(U));
            const volTensorField& gradU = tgradU();

            scalar val = 0;
            forAll(inputFields[idxI], cellI)
            {
                val = mag(U[cellI] & gradU[cellI]) / (mag(U[cellI] & U[cellI]) + mag(U[cellI] & gradU[cellI]) + 1e-16);
                inputFields[idxI][cellI] = (val + inputShift_[idxI]) * inputScale_[idxI];
            }
        }
        else if (inputName == "UOrth")
        {
            // Non-orthogonality between velocity and its gradient
            const volVectorField& U = mesh_.thisDb().lookupObject<volVectorField>("U");
            const tmp<volTensorField> tgradU(fvc::grad(U));
            const volTensorField& gradU = tgradU();

            scalar val = 0;
            forAll(inputFields[idxI], cellI)
            {
                val = mag(U[cellI] & gradU[cellI] & U[cellI]) / (mag(U[cellI]) * mag(gradU[cellI] & U[cellI]) + mag(U[cellI] & gradU[cellI] & U[cellI]) + 1e-16);
                inputFields[idxI][cellI] = (val + inputShift_[idxI]) * inputScale_[idxI];
            }
        }
        else if (inputName == "KoU2")
        {
            // turbulence intensity / velocity square
            const volScalarField& k = mesh_.thisDb().lookupObject<volScalarField>("k");
            const volVectorField& U = mesh_.thisDb().lookupObject<volVectorField>("U");
            scalar val = 0;
            forAll(inputFields[idxI], cellI)
            {
                val = k[cellI] / (0.5 * (U[cellI] & U[cellI]) + 1e-16);
                inputFields[idxI][cellI] = (val + inputShift_[idxI]) * inputScale_[idxI];
            }
        }
        else if (inputName == "ReWall")
        {
            // wall distance based Reynolds number
            const volScalarField& y = mesh_.thisDb().lookupObject<volScalarField>("yWall");
            const volScalarField& k = mesh_.thisDb().lookupObject<volScalarField>("k");
            volScalarField nu = daModel_.getDATurbulenceModel().nu();
            scalar val = 0;
            forAll(inputFields[idxI], cellI)
            {
                val = sqrt(k[cellI]) * y[cellI] / (50.0 * nu[cellI]);
                inputFields[idxI][cellI] = (val + inputShift_[idxI]) * inputScale_[idxI];
            }
        }
        else if (inputName == "CoP")
        {
            // convective / production
            daModel_.getTurbConvOverProd(inputFields[idxI]);
            forAll(inputFields[idxI], cellI)
            {
                inputFields[idxI][cellI] = (inputFields[idxI][cellI] + inputShift_[idxI]) * inputScale_[idxI];
            }
        }
        else if (inputName == "TauoK")
        {
            // ratio of total to normal Reynolds stress
            const volScalarField& k = mesh_.thisDb().lookupObject<volScalarField>("k");
            const volScalarField& nut = mesh_.thisDb().lookupObject<volScalarField>("nut");
            const volVectorField& U = mesh_.thisDb().lookupObject<volVectorField>("U");
            volSymmTensorField tau(2.0 / 3.0 * I * k - nut * twoSymm(fvc::grad(U)));
            scalar val = 0;
            forAll(inputFields[idxI], cellI)
            {
                val = mag(tau[cellI]) / (k[cellI] + 1e-16);
                inputFields[idxI][cellI] = (val + inputShift_[idxI]) * inputScale_[idxI];
            }
        }
        else
        {
            FatalErrorIn("") << "inputName: " << inputName << " not supported. Options are: VoS, PoD, chiSA, pGradStream, PSoSS, SCurv, UOrth, KoU2, ReWall, CoP, TauoK" << abort(FatalError);
        }
    }
}

label DARegression::compute()
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

        Return 1 if there is invalid value in the output. Return 0 if successful

    Output:
        a volScalarField prescribed by outputName
    */

    if (!active_)
    {
        return 0;
    }

    label fail = 0;

    volScalarField& outputField = const_cast<volScalarField&>(mesh_.thisDb().lookupObject<volScalarField>(outputName_));

    if (modelType_ == "neuralNetwork")
    {
        label nHiddenLayers = hiddenLayerNeurons_.size();

        List<List<scalar>> inputFields;
        inputFields.setSize(inputNames_.size());

        this->calcInput(inputFields);

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
                        // for the 1st hidden layer, we use the input layer as the input
                        forAll(inputNames_, neuronJ)
                        {
                            // weighted sum
                            layerVals[layerI][neuronI] += inputFields[neuronJ][cellI] * parameters_[counterI];
                            counterI++;
                        }
                    }
                    else
                    {
                        // for the rest of hidden layer, we use the previous hidden layer as the input
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
                    if (activationFunction_ == "sigmoid")
                    {
                        layerVals[layerI][neuronI] = 1 / (1 + exp(-layerVals[layerI][neuronI]));
                    }
                    else if (activationFunction_ == "tanh")
                    {
                        layerVals[layerI][neuronI] = (1 - exp(-2 * layerVals[layerI][neuronI])) / (1 + exp(-2 * layerVals[layerI][neuronI]));
                    }
                    else if (activationFunction_ == "ReLU")
                    {
                        if (layerVals[layerI][neuronI] < 0)
                        {
                            layerVals[layerI][neuronI] = leakyCoeff_ * layerVals[layerI][neuronI];
                        }
                    }
                    else
                    {
                        FatalErrorIn("") << "activationFunction not valid. Options are: sigmoid, tanh, and ReLU" << abort(FatalError);
                    }
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

            // no activation function for the output layer

            outputField[cellI] = outputScale_ * (outputVal + outputShift_);
        }

        // check if the output values are valid otherwise fix/bound them
        fail = this->checkOutput(outputField);

        outputField.correctBoundaryConditions();
    }
    else if (modelType_ == "radialBasisFunction")
    {
        List<List<scalar>> inputFields;
        inputFields.setSize(inputNames_.size());

        this->calcInput(inputFields);

        label nInputs = inputNames_.size();

        // increment of the parameters for each RBF basis
        label dP = 2 * nInputs + 1;

        forAll(mesh_.cells(), cellI)
        {
            scalar outputVal = 0.0;
            for (label i = 0; i < nRBFs_; i++)
            {
                scalar expCoeff = 0.0;
                for (label j = 0; j < nInputs; j++)
                {
                    scalar A = (inputFields[j][cellI] - parameters_[dP * i + 2 * j]) * (inputFields[j][cellI] - parameters_[dP * i + 2 * j]);
                    scalar B = 2 * parameters_[dP * i + 2 * j + 1] * parameters_[dP * i + 2 * j + 1];
                    expCoeff += A / B;
                }
                outputVal += parameters_[(dP + 1) * i + dP] * exp(-expCoeff);
            }

            outputField[cellI] = outputScale_ * (outputVal + outputShift_);
        }

        // check if the output values are valid otherwise fix/bound them
        fail = this->checkOutput(outputField);

        outputField.correctBoundaryConditions();
    }
    else
    {
        FatalErrorIn("") << "modelType_: " << modelType_ << " not supported. Options are: neuralNetwork and radialBasisFunction" << abort(FatalError);
    }

    return fail;
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
    else if (modelType_ == "radialBasisFunction")
    {
        label nInputs = inputNames_.size();

        // each RBF has a weight, nInputs mean, and nInputs std
        label nParameters = nRBFs_ * (2 * nInputs + 1);

        return nParameters;
    }
    else
    {
        FatalErrorIn("") << "modelType_: " << modelType_ << " not supported. Options are: neuralNetwork and radialBasisFunction" << abort(FatalError);
    }
}

label DARegression::checkOutput(volScalarField& outputField)
{
    /*
    Description:
        check if the output values are valid otherwise bound or fix them

        Output:

            Return 1 if there is invalid value in the output. Return 0 if successful
    */

    label fail = 0;

    // check if the output value is valid.
    label isNaN = 0;
    label isInf = 0;
    label isBounded = 0;
    forAll(mesh_.cells(), cellI)
    {
        if (std::isnan(outputField[cellI]))
        {
            outputField[cellI] = defaultOutputValue_;
            isNaN = 1;
        }
        if (std::isinf(outputField[cellI]))
        {
            outputField[cellI] = defaultOutputValue_;
            isInf = 1;
        }
        if (outputField[cellI] > outputUpperBound_)
        {
            outputField[cellI] = outputUpperBound_;
            isBounded = 1;
        }
        if (outputField[cellI] < outputLowerBound_)
        {
            outputField[cellI] = outputLowerBound_;
            isBounded = 1;
        }
    }
    if (isBounded == 1)
    {
        Info << "************* Warning! output values are bounded between " << outputLowerBound_ << " and " << outputUpperBound_ << endl;
        fail = 1;
    }

    if (isNaN == 1)
    {
        Info << "************* Warning! output values have nan and are set to " << defaultOutputValue_ << endl;
        fail = 1;
    }

    if (isInf == 1)
    {
        Info << "************* Warning! output values have inf and are set to " << defaultOutputValue_ << endl;
        fail = 1;
    }

    reduce(fail, sumOp<label>());

    return fail;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
