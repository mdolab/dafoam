/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v2

\*---------------------------------------------------------------------------*/

#include "DAIntmdVar.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DAIntmdVar::DAIntmdVar(
    const fvMesh& mesh,
    const DAOption& daOption)
    : mesh_(mesh),
      daOption_(daOption),
      volScalarList_(20), // reserve 20 fields
      volVectorList_(20)
{
    const dictionary intmdVarSubDict = daOption_.getAllOptions().subDict("intmdVar");
    if (intmdVarSubDict.size() != 0)
    {
        hasIntmdVar_ = 1;
        forAll(intmdVarSubDict.toc(), idxI)
        {
            word varName = intmdVarSubDict.toc()[idxI];
            dictionary varSubDict = intmdVarSubDict.subDict(varName);
            word operation = varSubDict.get<word>("operation");

            if (operation == "Mean")
            {
                word fieldType = varSubDict.get<word>("fieldType");
                word baseField = varSubDict.get<word>("baseField");
                word intmdFieldName = baseField + operation;
                if (fieldType == "volScalarField")
                {
                    const volScalarField& baseState = mesh_.thisDb().lookupObject<volScalarField>(baseField);
                    volScalarList_.set(
                        idxI,
                        new volScalarField(
                            IOobject(
                                intmdFieldName,
                                mesh_.time().timeName(),
                                mesh_,
                                IOobject::NO_READ,
                                IOobject::AUTO_WRITE), // do not register to db
                            baseState)
                    );
                }
                else if (fieldType == "volVectorField")
                {
                    const volVectorField& baseState = mesh_.thisDb().lookupObject<volVectorField>(baseField);
                    volVectorList_.set(
                        idxI,
                        new volVectorField(
                            IOobject(
                                intmdFieldName,
                                mesh_.time().timeName(),
                                mesh_,
                                IOobject::NO_READ,
                                IOobject::AUTO_WRITE), // do not register to db
                            baseState)
                    );
                }
                else
                {
                    FatalErrorIn("") << "fieldType not supported!"
                                     << abort(FatalError);
                }
            }
            else
            {
                FatalErrorIn("") << "operation not supported!"
                                 << abort(FatalError);
            }
        }
    }
}

void DAIntmdVar::update()
{
    /*
    Description:
        Update the values for all intermediate variables defined in DAOption-intmdVar

    */

    if (hasIntmdVar_ != 0)
    {
        const dictionary intmdVarSubDict = daOption_.getAllOptions().subDict("intmdVar");

        forAll(intmdVarSubDict.toc(), idxI)
        {
            word varName = intmdVarSubDict.toc()[idxI];
            dictionary varSubDict = intmdVarSubDict.subDict(varName);
            word operation = varSubDict.get<word>("operation");

            if (operation == "Mean")
            {
                // Modified based on
                // src/functionObjects/field/fieldAverage/fieldAverageItem/fieldAverageItemTemplates.C
                word fieldType = varSubDict.get<word>("fieldType");
                word baseField = varSubDict.get<word>("baseField");
                label restartSteps = varSubDict.get<label>("restartSteps");
                label nIdx = mesh_.time().timeIndex() % restartSteps;
                if (fieldType == "volScalarField")
                {
                    const volScalarField& baseState = mesh_.thisDb().lookupObject<volScalarField>(baseField);
                    volScalarField& intmdField = volScalarList_[idxI];
                    if (nIdx != 0)
                    {
                        // no need to compute Mean for nIdx = 0 because it will be assigned baseState 
                        // when restart begines, i.e., nIdx = 1
                        scalar beta = 1.0 / nIdx;
                        intmdField = beta * baseState + (1.0 - beta) * intmdField;
                    }
                }
                else if (fieldType == "volVectorField")
                {
                    const volVectorField& baseState = mesh_.thisDb().lookupObject<volVectorField>(baseField);
                    volVectorField& intmdField = volVectorList_[idxI];
                    if (nIdx != 0)
                    {
                        // no need to compute Mean for nIdx = 0 because it will be assigned baseState  
                        // when restart begines, i.e., nIdx = 1
                        scalar beta = 1.0 / nIdx;
                        intmdField = beta * baseState + (1.0 - beta) * intmdField;
                    }
                }
            }
        }
    }
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
