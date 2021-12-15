/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v2

\*---------------------------------------------------------------------------*/

#include "DAResidual.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

defineTypeNameAndDebug(DAResidual, 0);
defineRunTimeSelectionTable(DAResidual, dictionary);

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DAResidual::DAResidual(
    const word modelType,
    const fvMesh& mesh,
    const DAOption& daOption,
    const DAModel& daModel,
    const DAIndex& daIndex)
    : mesh_(mesh),
      daOption_(daOption),
      daModel_(daModel),
      daIndex_(daIndex),
      daField_(mesh, daOption, daModel, daIndex)
{
}

// * * * * * * * * * * * * * * * * * Selectors * * * * * * * * * * * * * * * //

autoPtr<DAResidual> DAResidual::New(
    const word modelType,
    const fvMesh& mesh,
    const DAOption& daOption,
    const DAModel& daModel,
    const DAIndex& daIndex)
{
    // standard setup for runtime selectable classes

    if (daOption.getAllOptions().lookupOrDefault<label>("debug", 0))
    {
        Info << "Selecting " << modelType << " for DAResidual" << endl;
    }

    dictionaryConstructorTable::iterator cstrIter =
        dictionaryConstructorTablePtr_->find(modelType);

    // if the solver name is not found in any child class, print an error
    if (cstrIter == dictionaryConstructorTablePtr_->end())
    {
        FatalErrorIn(
            "DAResidual::New"
            "("
            "    const word,"
            "    const fvMesh&,"
            "    const DAOption&,"
            "    const DAModel&,"
            "    const DAIndex&"
            ")")
            << "Unknown DAResidual type "
            << modelType << nl << nl
            << "Valid DAResidual types:" << endl
            << dictionaryConstructorTablePtr_->sortedToc()
            << exit(FatalError);
    }

    // child class found
    return autoPtr<DAResidual>(
        cstrIter()(modelType, mesh, daOption, daModel, daIndex));
}

// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

void DAResidual::masterFunction(
    const dictionary& options,
    const Vec xvVec,
    const Vec wVec,
    Vec resVec)
{
    /*
    Description:
        A master function that takes the volume mesh points and state variable vecs
        as input, and compute the residual vector.
    
    Input:
        options.updateState: whether to assign the values in wVec to the state
        variables of the OpenFOAM fields (e.g., U, p). This will also update boundary conditions
        and update all intermediate variables that are dependent on the state 
        variables. 

        options.updateMesh: whether to assign the values in xvVec to the OpenFOAM mesh 
        coordinates in Foam::fvMesh. This will also call mesh.movePoints() to update
        all the mesh metrics such as mesh volume, cell centers, mesh surface area, etc.

        options.setResVec: whether to assign the residuals (e.g., URes_) from the OpenFOAM fields
        to the resVec. If set to 0, nothing will be written to resVec

        options.isPC: whether to compute residual for constructing PC matrix

        xvVec: the volume coordinates vector (flatten)

        wVec: the state variable vector
    
    Output:
        resVec: the residual vector

    NOTE1: the ordering of the xvVec, wVec, and resVec depends on adjStateOrdering, see 
    Foam::DAIndex for details

    NOTE2: the calcResiduals function will be implemented in the child classes
    */

    VecZeroEntries(resVec);

    DAModel& daModel = const_cast<DAModel&>(daModel_);

    label updateState = options.getLabel("updateState");

    label updateMesh = options.getLabel("updateMesh");

    label setResVec = options.getLabel("setResVec");

    if (updateMesh)
    {
        daField_.pointVec2OFMesh(xvVec);
    }

    if (updateState)
    {
        daField_.stateVec2OFField(wVec);

        // now update intermediate states and boundry conditions
        this->correctBoundaryConditions();
        this->updateIntermediateVariables();
        daModel.correctBoundaryConditions();
        daModel.updateIntermediateVariables();
        // if there are special boundary conditions, apply special treatment
        daField_.specialBCTreatment();
    }

    this->calcResiduals(options);
    daModel.calcResiduals(options);

    if (setResVec)
    {
        // asssign the openfoam residual field to resVec
        daField_.ofResField2ResVec(resVec);
    }
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
