/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

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
    : regIOobject(
        IOobject(
            "DARegression", // the db name
            mesh.time().timeName(),
            mesh, // register to mesh
            IOobject::NO_READ,
            IOobject::NO_WRITE,
            true // always register object
            )),
      mesh_(mesh),
      daOption_(daOption),
      daModel_(daModel)
{
    dictionary regSubDict = daOption.getAllOptions().subDict("regressionModel");
    active_ = regSubDict.getLabel("active");

    // initialize parameters
    if (active_)
    {
        forAll(regSubDict.toc(), idxI)
        {
            word key = regSubDict.toc()[idxI];
            if (key != "active")
            {
                modelNames_.append(key);
            }
        }

        forAll(modelNames_, idxI)
        {
            word modelName = modelNames_[idxI];
            dictionary modelSubDict = daOption.getAllOptions().subDict("regressionModel").subDict(modelName);

            modelType_.set(modelName, modelSubDict.getWord("modelType"));

            wordList tempWordList;
            scalarList tempScalarList;
            labelList tempLabelList;

            modelSubDict.readEntry<wordList>("inputNames", tempWordList);
            inputNames_.set(modelName, tempWordList);

            outputName_.set(modelName, modelSubDict.getWord("outputName"));

            modelSubDict.readEntry<scalarList>("inputShift", tempScalarList);
            inputShift_.set(modelName, tempScalarList);

            modelSubDict.readEntry<scalarList>("inputScale", tempScalarList);
            inputScale_.set(modelName, tempScalarList);

            outputShift_.set(modelName, modelSubDict.getScalar("outputShift"));

            outputScale_.set(modelName, modelSubDict.getScalar("outputScale"));

            outputUpperBound_.set(modelName, modelSubDict.getScalar("outputUpperBound"));

            outputLowerBound_.set(modelName, modelSubDict.getScalar("outputLowerBound"));

            defaultOutputValue_.set(modelName, modelSubDict.getScalar("defaultOutputValue"));

            printInputInfo_.set(modelName, modelSubDict.getLabel("printInputInfo"));

            writeFeatures_.set(modelName, modelSubDict.lookupOrDefault<label>("writeFeatures", 0));

            if (modelType_[modelName] == "neuralNetwork")
            {
                modelSubDict.readEntry<labelList>("hiddenLayerNeurons", tempLabelList);
                hiddenLayerNeurons_.set(modelName, tempLabelList);
                activationFunction_.set(modelName, modelSubDict.getWord("activationFunction"));
                if (activationFunction_[modelName] == "relu")
                {
                    leakyCoeff_.set(modelName, modelSubDict.lookupOrDefault<scalar>("leakyCoeff", 0.0));
                }
            }
            else if (modelType_[modelName] == "radialBasisFunction")
            {
                nRBFs_.set(modelName, modelSubDict.getLabel("nRBFs"));
            }
            else if (modelType_[modelName] == "externalTensorFlow")
            {
                useExternalModel_ = 1;
                // here the array size is chosen based on the regModel that has the largest number of inputs
                // this is important because we support multiple regModels but we don't want to create multiple featuresFlattenArray_
                if (inputNames_[modelName].size() * mesh_.nCells() > featuresFlattenArraySize_)
                {
                    featuresFlattenArraySize_ = inputNames_[modelName].size() * mesh_.nCells();
                }
            }
            else
            {
                FatalErrorIn("DARegression") << "modelType_: " << modelType_[modelName] << " not supported. Options are: neuralNetwork, radialBasisFunction, and externalTensorFlow" << abort(FatalError);
            }

            // check the sizes
            if (inputNames_[modelName].size() != inputShift_[modelName].size()
                || inputNames_[modelName].size() != inputScale_[modelName].size())
            {
                FatalErrorIn("DARegression") << "inputNames has different sizes than inputShift or inputScale" << abort(FatalError);
            }

            label nParameters = this->nParameters(modelName);
            scalarList parameters(nParameters, 0.0);
            parameters_.set(modelName, parameters);

            // initialize the ptr scalarFields
            label nInputs = inputNames_[modelName].size();
            PtrList<volScalarField> features(nInputs);
            forAll(inputNames_[modelName], idxI)
            {
                word inputName = inputNames_[modelName][idxI];
                features.set(
                    idxI,
                    new volScalarField(
                        IOobject(
                            inputName,
                            mesh_.time().timeName(),
                            mesh_,
                            IOobject::NO_READ,
                            IOobject::NO_WRITE),
                        mesh_,
                        dimensionedScalar(inputName, dimensionSet(0, 0, 0, 0, 0, 0, 0), 0.0)));
            }
            features_.set(modelName, features);
        }

        // if external model is used, initialize space for the features and output arrays
        if (useExternalModel_)
        {

#if defined(CODI_ADF)
            featuresFlattenArrayDouble_ = new double[featuresFlattenArraySize_];
            outputFieldArrayDouble_ = new double[mesh_.nCells()];
#else
            featuresFlattenArray_ = new scalar[featuresFlattenArraySize_];
            outputFieldArray_ = new scalar[mesh_.nCells()];
#endif
        }
    }
}

void DARegression::calcInputFeatures(word modelName)
{
    /*
    Description:
        Calculate the input features. Here features is a unique list for all the regModel's inputNames.
        This is because two regModel's inputNames can have overlap, so we don't want to compute
        duplicated features

        NOTE: if a feature is a ratio between variable A and variable B, we will normalize 
        it such that the range of this feature is from -1 to 1 by using:
        feature = A / (A + B + 1e-16)
    */

    forAll(features_[modelName], idxI)
    {
        word inputName = inputNames_[modelName][idxI];
        if (inputName == "VoS")
        {
            // vorticity / strain
            const volVectorField& U = mesh_.thisDb().lookupObject<volVectorField>("U");
            const tmp<volTensorField> tgradU(fvc::grad(U));
            const volTensorField& gradU = tgradU();
            volScalarField magOmega = mag(skew(gradU));
            volScalarField magS = mag(symm(gradU));
            forAll(features_[modelName][idxI], cellI)
            {
                features_[modelName][idxI][cellI] = (magOmega[cellI] / (magS[cellI] + magOmega[cellI] + 1e-16) + inputShift_[modelName][idxI]) * inputScale_[modelName][idxI];
            }
            features_[modelName][idxI].correctBoundaryConditions();
        }
        else if (inputName == "PoD")
        {
            // production / destruction
            daModel_.getTurbProdOverDestruct(features_[modelName][idxI]);
            forAll(features_[modelName][idxI], cellI)
            {
                features_[modelName][idxI][cellI] = (features_[modelName][idxI][cellI] + inputShift_[modelName][idxI]) * inputScale_[modelName][idxI];
            }
            features_[modelName][idxI].correctBoundaryConditions();
        }
        else if (inputName == "chiSA")
        {
            // the chi() function from SA
            const volScalarField& nuTilda = mesh_.thisDb().lookupObject<volScalarField>("nuTilda");
            volScalarField nu = daModel_.getDATurbulenceModel().nu();
            forAll(features_[modelName][idxI], cellI)
            {
                features_[modelName][idxI][cellI] = (nuTilda[cellI] / (nu[cellI] + nuTilda[cellI] + 1e-16) + inputShift_[modelName][idxI]) * inputScale_[modelName][idxI];
            }
            features_[modelName][idxI].correctBoundaryConditions();
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
            volScalarField::Internal pGradAlongStream = (U() & pGrad()) / pG_denominator();

            forAll(features_[modelName][idxI], cellI)
            {
                features_[modelName][idxI][cellI] = (pGradAlongStream[cellI] + inputShift_[modelName][idxI]) * inputScale_[modelName][idxI];
            }
            features_[modelName][idxI].correctBoundaryConditions();
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
            forAll(features_[modelName][idxI], cellI)
            {
                diagUGrad[0] = gradU[cellI].xx();
                diagUGrad[1] = gradU[cellI].yy();
                diagUGrad[2] = gradU[cellI].zz();
                val = mag(pGrad[cellI]) / (mag(pGrad[cellI]) + mag(3.0 * cmptAv(U[cellI] & diagUGrad)) + 1e-16);
                features_[modelName][idxI][cellI] = (val + inputShift_[modelName][idxI]) * inputScale_[modelName][idxI];
            }
            features_[modelName][idxI].correctBoundaryConditions();
        }
        else if (inputName == "SCurv")
        {
            // streamline curvature
            const volVectorField& U = mesh_.thisDb().lookupObject<volVectorField>("U");
            const tmp<volTensorField> tgradU(fvc::grad(U));
            const volTensorField& gradU = tgradU();

            scalar val = 0;
            forAll(features_[modelName][idxI], cellI)
            {
                val = mag(U[cellI] & gradU[cellI]) / (mag(U[cellI] & U[cellI]) + mag(U[cellI] & gradU[cellI]) + 1e-16);
                features_[modelName][idxI][cellI] = (val + inputShift_[modelName][idxI]) * inputScale_[modelName][idxI];
            }
            features_[modelName][idxI].correctBoundaryConditions();
        }
        else if (inputName == "UOrth")
        {
            // Non-orthogonality between velocity and its gradient
            const volVectorField& U = mesh_.thisDb().lookupObject<volVectorField>("U");
            const tmp<volTensorField> tgradU(fvc::grad(U));
            const volTensorField& gradU = tgradU();

            scalar val = 0;
            forAll(features_[modelName][idxI], cellI)
            {
                val = mag(U[cellI] & gradU[cellI] & U[cellI]) / (mag(U[cellI]) * mag(gradU[cellI] & U[cellI]) + mag(U[cellI] & gradU[cellI] & U[cellI]) + 1e-16);
                features_[modelName][idxI][cellI] = (val + inputShift_[modelName][idxI]) * inputScale_[modelName][idxI];
            }
            features_[modelName][idxI].correctBoundaryConditions();
        }
        else if (inputName == "KoU2")
        {
            // turbulence intensity / velocity square
            const volScalarField& k = mesh_.thisDb().lookupObject<volScalarField>("k");
            const volVectorField& U = mesh_.thisDb().lookupObject<volVectorField>("U");
            scalar val = 0;
            forAll(features_[modelName][idxI], cellI)
            {
                val = k[cellI] / (0.5 * (U[cellI] & U[cellI]) + k[cellI] + 1e-16);
                features_[modelName][idxI][cellI] = (val + inputShift_[modelName][idxI]) * inputScale_[modelName][idxI];
            }
            features_[modelName][idxI].correctBoundaryConditions();
        }
        else if (inputName == "ReWall")
        {
            // wall distance based Reynolds number
            const volScalarField& y = mesh_.thisDb().lookupObject<volScalarField>("yWall");
            const volScalarField& k = mesh_.thisDb().lookupObject<volScalarField>("k");
            volScalarField nu = daModel_.getDATurbulenceModel().nu();
            scalar val = 0;
            forAll(features_[modelName][idxI], cellI)
            {
                val = sqrt(k[cellI]) * y[cellI] / (50.0 * nu[cellI] + sqrt(k[cellI]) * y[cellI] + 1e-16);
                features_[modelName][idxI][cellI] = (val + inputShift_[modelName][idxI]) * inputScale_[modelName][idxI];
            }
            features_[modelName][idxI].correctBoundaryConditions();
        }
        else if (inputName == "CoP")
        {
            // convective / production
            daModel_.getTurbConvOverProd(features_[modelName][idxI]);
            forAll(features_[modelName][idxI], cellI)
            {
                features_[modelName][idxI][cellI] = (features_[modelName][idxI][cellI] + inputShift_[modelName][idxI]) * inputScale_[modelName][idxI];
            }
            features_[modelName][idxI].correctBoundaryConditions();
        }
        else if (inputName == "TauoK")
        {
            // ratio of total to normal Reynolds stress
            const volScalarField& k = mesh_.thisDb().lookupObject<volScalarField>("k");
            const volScalarField& nut = mesh_.thisDb().lookupObject<volScalarField>("nut");
            const volVectorField& U = mesh_.thisDb().lookupObject<volVectorField>("U");
            volSymmTensorField tau(2.0 / 3.0 * I * k - nut * twoSymm(fvc::grad(U)));
            scalar val = 0;
            forAll(features_[modelName][idxI], cellI)
            {
                val = mag(tau[cellI]) / (k[cellI] + mag(tau[cellI]) + 1e-16);
                features_[modelName][idxI][cellI] = (val + inputShift_[modelName][idxI]) * inputScale_[modelName][idxI];
            }
            features_[modelName][idxI].correctBoundaryConditions();
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
            inputNames: a list of volScalarFields saved in the features_ pointerList
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

    forAll(modelNames_, idxI)
    {
        word modelName = modelNames_[idxI];

        // compute the inputFeature for all inputs
        this->calcInputFeatures(modelName);

        // if the output variable is not found in the Db, just return and do nothing
        if (!mesh_.thisDb().foundObject<volScalarField>(outputName_[modelName]))
        {
            return 0;
        }

        volScalarField& outputField = const_cast<volScalarField&>(mesh_.thisDb().lookupObject<volScalarField>(outputName_[modelName]));

        if (modelType_[modelName] == "neuralNetwork")
        {
            label nHiddenLayers = hiddenLayerNeurons_[modelName].size();
            List<List<scalar>> layerVals;
            layerVals.setSize(nHiddenLayers);
            for (label layerI = 0; layerI < nHiddenLayers; layerI++)
            {
                label nNeurons = hiddenLayerNeurons_[modelName][layerI];
                layerVals[layerI].setSize(nNeurons);
            }

            forAll(mesh_.cells(), cellI)
            {
                label counterI = 0;

                for (label layerI = 0; layerI < nHiddenLayers; layerI++)
                {
                    label nNeurons = hiddenLayerNeurons_[modelName][layerI];
                    forAll(layerVals[layerI], neuronI)
                    {
                        layerVals[layerI][neuronI] = 0.0;
                    }
                    for (label neuronI = 0; neuronI < nNeurons; neuronI++)
                    {
                        if (layerI == 0)
                        {
                            // for the 1st hidden layer, we use the input layer as the input
                            forAll(inputNames_[modelName], neuronJ)
                            {
                                // weighted sum
                                layerVals[layerI][neuronI] += features_[modelName][neuronJ][cellI] * parameters_[modelName][counterI];
                                counterI++;
                            }
                        }
                        else
                        {
                            // for the rest of hidden layer, we use the previous hidden layer as the input
                            forAll(layerVals[layerI - 1], neuronJ)
                            {
                                // weighted sum
                                layerVals[layerI][neuronI] += layerVals[layerI - 1][neuronJ] * parameters_[modelName][counterI];
                                counterI++;
                            }
                        }
                        // bias
                        layerVals[layerI][neuronI] += parameters_[modelName][counterI];
                        counterI++;
                        // activation function
                        if (activationFunction_[modelName] == "sigmoid")
                        {
                            layerVals[layerI][neuronI] = 1 / (1 + exp(-layerVals[layerI][neuronI]));
                        }
                        else if (activationFunction_[modelName] == "tanh")
                        {
                            layerVals[layerI][neuronI] = (1 - exp(-2 * layerVals[layerI][neuronI])) / (1 + exp(-2 * layerVals[layerI][neuronI]));
                        }
                        else if (activationFunction_[modelName] == "relu")
                        {
                            if (layerVals[layerI][neuronI] < 0)
                            {
                                layerVals[layerI][neuronI] = leakyCoeff_[modelName] * layerVals[layerI][neuronI];
                            }
                        }
                        else
                        {
                            FatalErrorIn("") << "activationFunction not valid. Options are: sigmoid, tanh, and relu" << abort(FatalError);
                        }
                    }
                }
                // final output layer, we have only one output
                scalar outputVal = 0.0;
                forAll(layerVals[nHiddenLayers - 1], neuronJ)
                {
                    // weighted sum
                    outputVal += layerVals[nHiddenLayers - 1][neuronJ] * parameters_[modelName][counterI];
                    counterI++;
                }
                // bias
                outputVal += parameters_[modelName][counterI];

                // no activation function for the output layer

                outputField[cellI] = outputScale_[modelName] * (outputVal + outputShift_[modelName]);
            }

            // check if the output values are valid otherwise fix/bound them
            fail += this->checkOutput(modelName, outputField);

            outputField.correctBoundaryConditions();
        }
        else if (modelType_[modelName] == "radialBasisFunction")
        {
            label nInputs = inputNames_[modelName].size();

            // increment of the parameters for each RBF basis
            label dP = 2 * nInputs + 1;

            forAll(mesh_.cells(), cellI)
            {
                scalar outputVal = 0.0;
                for (label i = 0; i < nRBFs_[modelName]; i++)
                {
                    scalar expCoeff = 0.0;
                    for (label j = 0; j < nInputs; j++)
                    {
                        scalar A = (features_[modelName][j][cellI] - parameters_[modelName][dP * i + 2 * j]) * (features_[modelName][j][cellI] - parameters_[modelName][dP * i + 2 * j]);
                        scalar B = 2 * parameters_[modelName][dP * i + 2 * j + 1] * parameters_[modelName][dP * i + 2 * j + 1];
                        expCoeff += A / B;
                    }
                    outputVal += parameters_[modelName][dP * i + dP - 1] * exp(-expCoeff);
                }

                outputField[cellI] = outputScale_[modelName] * (outputVal + outputShift_[modelName]);
            }

            // check if the output values are valid otherwise fix/bound them
            fail += this->checkOutput(modelName, outputField);

            outputField.correctBoundaryConditions();
        }
        else if (modelType_[modelName] == "externalTensorFlow")
        {
            label nInputs = inputNames_[modelName].size();
            label nCells = mesh_.nCells();

            DAUtility::pySetModelNameInterface(modelName.c_str(), DAUtility::pySetModelName);

            // NOTE: forward mode not supported..
#if defined(CODI_ADR)
            Info << "WARNINGN...... Regression model for ADR is not implemented..." << endl;
            /*
            // assign features_ to featuresFlattenArray_
            // here featuresFlattenArray_ should be order like this to facilitate Python layer reshape:
            // [(cell1, feature1), (cell1, feature2), ... (cell2, feature1), (cell2, feature2) ... ]
            label counterI = 0;
            // loop over all features
            forAll(features_[modelName], idxI)
            {
                // loop over all cells
                forAll(features_[modelName][idxI], cellI)
                {
                    counterI = cellI * nCells + idxI;
                    featuresFlattenArray_[counterI] = features_[modelName][idxI][cellI];
                }
            }
            // assign outputField to outputFieldArray_
            forAll(outputField, cellI)
            {
                outputFieldArray_[cellI] = outputField[cellI];
            }

            // we need to use the external function helper from CoDiPack to propagate the AD
            codi::ExternalFunctionHelper<codi::RealReverse> externalFunc;
            for (label i = 0; i < mesh_.nCells() * nInputs; i++)
            {
                externalFunc.addInput(featuresFlattenArray_[i]);
            }

            for (label i = 0; i < mesh_.nCells(); i++)
            {
                externalFunc.addOutput(outputFieldArray_[i]);
            }

            externalFunc.callPrimalFunc(DARegression::betaCompute);

            codi::RealReverse::Tape& tape = codi::RealReverse::getTape();

            if (tape.isActive())
            {
                externalFunc.addToTape(DARegression::betaJacVecProd);
            }

            forAll(outputField, cellI)
            {
                outputField[cellI] = outputFieldArray_[cellI];
            }
            */

#elif defined(CODI_ADF)
            Info << "WARNINGN...... Regression model for ADF is not implemented..." << endl;
            /*
            // assign features_ to featuresFlattenArray_
            // here featuresFlattenArray_ should be order like this to facilitate Python layer reshape:
            // [(cell1, feature1), (cell1, feature2), ... (cell2, feature1), (cell2, feature2) ... ]
            label counterI = 0;
            // loop over all features
            forAll(features_[modelName], idxI)
            {
                // loop over all cells
                forAll(features_[modelName][idxI], cellI)
                {
                    counterI = cellI * nCells + idxI;
                    featuresFlattenArrayDouble_[counterI] = features_[modelName][idxI][cellI].getValue();
                }
            }
            // assign outputField to outputFieldArrayDouble_
            forAll(outputField, cellI)
            {
                outputFieldArrayDouble_[cellI] = outputField[cellI].value();
            }

            // python callback function
            DAUtility::pyCalcBetaInterface(
                featuresFlattenArrayDouble_, mesh_.nCells() * nInputs, outputFieldArrayDouble_, mesh_.nCells(), DAUtility::pyCalcBeta);

            forAll(outputField, cellI)
            {
                outputField[cellI] = outputFieldArrayDouble_[cellI];
            }
            */

#else
            // assign features_ to featuresFlattenArray_
            // here featuresFlattenArray_ should be order like this to facilitate Python layer reshape:
            // [(cell1, feature1), (cell1, feature2), ... (cell2, feature1), (cell2, feature2) ... ]
            label counterI = 0;
            // loop over all features
            forAll(features_[modelName], idxI)
            {
                // loop over all cells
                forAll(features_[modelName][idxI], cellI)
                {
                    counterI = cellI * nInputs + idxI;
                    featuresFlattenArray_[counterI] = features_[modelName][idxI][cellI];
                }
            }
            // assign outputField to outputFieldArray_
            forAll(outputField, cellI)
            {
                outputFieldArray_[cellI] = outputField[cellI];
            }

            // python callback function
            DAUtility::pyCalcBetaInterface(
                featuresFlattenArray_, mesh_.nCells() * nInputs, outputFieldArray_, mesh_.nCells(), DAUtility::pyCalcBeta);

            forAll(outputField, cellI)
            {
                outputField[cellI] = outputFieldArray_[cellI];
            }
#endif
        }
        else
        {
            FatalErrorIn("") << "modelType_: " << modelType_ << " not supported. Options are: neuralNetwork, radialBasisFunction, and externalTensorFlow" << abort(FatalError);
        }
    }

    return fail;
}

label DARegression::nParameters(word modelName)
{
    /*
    Description:
        get the number of parameters
    */

    if (!active_)
    {
        FatalErrorIn("") << "nParameters() is called but the regression model is not active!" << abort(FatalError);
    }

    if (modelType_[modelName] == "neuralNetwork")
    {
        label nHiddenLayers = hiddenLayerNeurons_[modelName].size();
        label nInputs = inputNames_[modelName].size();

        // add weights
        // input
        label nParameters = nInputs * hiddenLayerNeurons_[modelName][0];
        // hidden layers
        for (label layerI = 1; layerI < hiddenLayerNeurons_[modelName].size(); layerI++)
        {
            nParameters += hiddenLayerNeurons_[modelName][layerI] * hiddenLayerNeurons_[modelName][layerI - 1];
        }
        // output
        nParameters += hiddenLayerNeurons_[modelName][nHiddenLayers - 1] * 1;

        // add biases
        // add hidden layers
        for (label layerI = 0; layerI < hiddenLayerNeurons_[modelName].size(); layerI++)
        {
            nParameters += hiddenLayerNeurons_[modelName][layerI];
        }
        // add output layer
        nParameters += 1;

        return nParameters;
    }
    else if (modelType_[modelName] == "radialBasisFunction")
    {
        label nInputs = inputNames_[modelName].size();

        // each RBF has a weight, nInputs mean, and nInputs std
        label nParameters = nRBFs_[modelName] * (2 * nInputs + 1);

        return nParameters;
    }
    else if (modelType_[modelName] == "externalTensorFlow")
    {
        // do nothing
        return 0;
    }
    else
    {
        FatalErrorIn("") << "modelType_: " << modelType_[modelName] << " not supported. Options are: neuralNetwork, radialBasisFunction, and externalTensorFlow" << abort(FatalError);
        return 0;
    }
}

label DARegression::checkOutput(word modelName, volScalarField& outputField)
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
            outputField[cellI] = defaultOutputValue_[modelName];
            isNaN = 1;
        }
        if (std::isinf(outputField[cellI]))
        {
            outputField[cellI] = defaultOutputValue_[modelName];
            isInf = 1;
        }
        if (outputField[cellI] > outputUpperBound_[modelName])
        {
            outputField[cellI] = outputUpperBound_[modelName];
            isBounded = 1;
        }
        if (outputField[cellI] < outputLowerBound_[modelName])
        {
            outputField[cellI] = outputLowerBound_[modelName];
            isBounded = 1;
        }
    }
    if (isBounded == 1)
    {
        Pout << "************* Warning! output values are bounded between " << outputLowerBound_[modelName] << " and " << outputUpperBound_[modelName] << endl;
        fail = 1;
    }

    if (isNaN == 1)
    {
        Pout << "************* Warning! output values have nan and are set to " << defaultOutputValue_[modelName] << endl;
        fail = 1;
    }

    if (isInf == 1)
    {
        Pout << "************* Warning! output values have inf and are set to " << defaultOutputValue_[modelName] << endl;
        fail = 1;
    }

    reduce(fail, sumOp<label>());

    return fail;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
