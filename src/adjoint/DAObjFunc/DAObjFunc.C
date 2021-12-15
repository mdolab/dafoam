/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v2

\*---------------------------------------------------------------------------*/

#include "DAObjFunc.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

defineTypeNameAndDebug(DAObjFunc, 0);
defineRunTimeSelectionTable(DAObjFunc, dictionary);

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DAObjFunc::DAObjFunc(
    const fvMesh& mesh,
    const DAOption& daOption,
    const DAModel& daModel,
    const DAIndex& daIndex,
    const DAResidual& daResidual,
    const word objFuncName,
    const word objFuncPart,
    const dictionary& objFuncDict)
    : mesh_(mesh),
      daOption_(daOption),
      daModel_(daModel),
      daIndex_(daIndex),
      daResidual_(daResidual),
      objFuncName_(objFuncName),
      objFuncPart_(objFuncPart),
      objFuncDict_(objFuncDict),
      daField_(mesh, daOption, daModel, daIndex)
{

    // calcualte the face and cell indices that are associated with this objective
    this->calcObjFuncSources(objFuncFaceSources_, objFuncCellSources_);

    // initialize objFuncFaceValues_ and objFuncCellValues_ and assign zeros
    // they will be computed later by calling DAObjFunc::calcObjFunc
    objFuncFaceValues_.setSize(objFuncFaceSources_.size());
    forAll(objFuncFaceValues_, idxI)
    {
        objFuncFaceValues_[idxI] = 0.0;
    }

    objFuncCellValues_.setSize(objFuncCellSources_.size());
    forAll(objFuncCellValues_, idxI)
    {
        objFuncCellValues_[idxI] = 0.0;
    }
}

// * * * * * * * * * * * * * * * * * Selectors * * * * * * * * * * * * * * * //

autoPtr<DAObjFunc> DAObjFunc::New(
    const fvMesh& mesh,
    const DAOption& daOption,
    const DAModel& daModel,
    const DAIndex& daIndex,
    const DAResidual& daResidual,
    const word objFuncName,
    const word objFuncPart,
    const dictionary& objFuncDict)
{
    // standard setup for runtime selectable classes

    // look up the solver name
    word modelType;
    objFuncDict.readEntry<word>("type", modelType);

    if (daOption.getAllOptions().lookupOrDefault<label>("debug", 0))
    {
        Info << "Selecting type: " << modelType << " for DAObjFunc. Name: " << objFuncName
             << " part: " << objFuncPart << endl;
    }

    dictionaryConstructorTable::iterator cstrIter =
        dictionaryConstructorTablePtr_->find(modelType);

    // if the solver name is not found in any child class, print an error
    if (cstrIter == dictionaryConstructorTablePtr_->end())
    {
        FatalErrorIn(
            "DAObjFunc::New"
            "("
            "    const fvMesh&,"
            "    const DAOption&,"
            "    const DAModel&,"
            "    const DAIndex&,"
            "    const DAResidual&,"
            "    const word,"
            "    const word,"
            "    const dictionary&"
            ")")
            << "Unknown DAObjFunc type "
            << modelType << nl << nl
            << "Valid DAObjFunc types:" << endl
            << dictionaryConstructorTablePtr_->sortedToc()
            << exit(FatalError);
    }

    // child class found
    return autoPtr<DAObjFunc>(
        cstrIter()(mesh,
                   daOption,
                   daModel,
                   daIndex,
                   daResidual,
                   objFuncName,
                   objFuncPart,
                   objFuncDict));
}

// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

void DAObjFunc::calcObjFuncSources(
    labelList& faceSources,
    labelList& cellSources)
{
    /*
    Description:
        Compute the face and cell sources for the objective function.

    Output:
        faceSources, cellSources: The face and cell indices that 
        are associated with this objective function

    Example:
        A typical objFunc dictionary reads:
    
        {
            "type": "force",
            "source": "patchToFace",
            "patches": ["walls", "wallsbump"],
            "scale": 0.5,
            "addToAdjoint": False
        }

        This information is obtained from DAObjFunc::objFuncDict_

    */

    // all avaiable source type are in src/meshTools/sets/cellSources
    // Example of IO parameters os in applications/utilities/mesh/manipulation/topoSet

    word objSource;
    objFuncDict_.readEntry("source", objSource);
    if (objSource == "patchToFace")
    {
        // create a topoSet
        autoPtr<topoSet> currentSet(
            topoSet::New(
                "faceSet",
                mesh_,
                "set0",
                IOobject::NO_READ));
        // create the source
        autoPtr<topoSetSource> sourceSet(
            topoSetSource::New(objSource, mesh_, objFuncDict_));

        // add the sourceSet to topoSet
        sourceSet().applyToSet(topoSetSource::NEW, currentSet());
        // get the face index from currentSet, we need to use
        // this special for loop
        for (const label i : currentSet())
        {
            faceSources.append(i);
        }
    }
    else if (objSource == "boxToCell")
    {
        // create a topoSet
        autoPtr<topoSet> currentSet(
            topoSet::New(
                "cellSet",
                mesh_,
                "set0",
                IOobject::NO_READ));
        // we need to change the min and max because they need to
        // be of type point; however, we can't parse point type
        // in pyDict, we need to change them here.
        dictionary objFuncTmp = objFuncDict_;
        scalarList boxMin;
        scalarList boxMax;
        objFuncDict_.readEntry("min", boxMin);
        objFuncDict_.readEntry("max", boxMax);

        point boxMin1;
        point boxMax1;
        boxMin1[0] = boxMin[0];
        boxMin1[1] = boxMin[1];
        boxMin1[2] = boxMin[2];
        boxMax1[0] = boxMax[0];
        boxMax1[1] = boxMax[1];
        boxMax1[2] = boxMax[2];

        objFuncTmp.set("min", boxMin1);
        objFuncTmp.set("max", boxMax1);

        // create the source
        autoPtr<topoSetSource> sourceSet(
            topoSetSource::New(objSource, mesh_, objFuncTmp));

        // add the sourceSet to topoSet
        sourceSet().applyToSet(topoSetSource::NEW, currentSet());
        // get the face index from currentSet, we need to use
        // this special for loop
        for (const label i : currentSet())
        {
            cellSources.append(i);
        }
    }
    else
    {
        FatalErrorIn("calcObjFuncSources") << "source: " << objSource << " not supported!"
                                           << "Options are: patchToFace, boxToCell!"
                                           << abort(FatalError);
    }
}

scalar DAObjFunc::masterFunction(
    const dictionary& options,
    const Vec xvVec,
    const Vec wVec)
{
    /*
    Description:
        A master function that takes the volume mesh points and state variable vecs
        as input, and compute the value of the objective and their discrete values
        on each face/cell source
    
    Input:
        options.updateState: whether to assign the values in wVec to the state
        variables of the OpenFOAM fields (e.g., U, p). This will also update boundary conditions
        and update all intermediate variables that are dependent on the state 
        variables. 

        options.updateMesh: whether to assign the values in xvVec to the OpenFOAM mesh 
        coordinates in Foam::fvMesh. This will also call mesh.movePoints() to update
        all the mesh metrics such as mesh volume, cell centers, mesh surface area, etc.

        xvVec: the volume coordinates vector (flatten)

        wVec: the state variable vector
    
    Output:
        objFuncValue: the reduced objective value

    */

    DAModel& daModel = const_cast<DAModel&>(daModel_);
    DAResidual& daResidual = const_cast<DAResidual&>(daResidual_);

    label updateState = 0;
    options.readEntry<label>("updateState", updateState);

    label updateMesh = 0;
    options.readEntry<label>("updateMesh", updateMesh);

    if (updateMesh)
    {
        daField_.pointVec2OFMesh(xvVec);
    }

    if (updateState)
    {
        daField_.stateVec2OFField(wVec);

        // now update intermediate states and boundry conditions
        daResidual.correctBoundaryConditions();
        daResidual.updateIntermediateVariables();
        daModel.correctBoundaryConditions();
        daModel.updateIntermediateVariables();
        // if there are special boundary conditions, apply special treatment
        daField_.specialBCTreatment();
    }

    scalar objFuncValue = this->getObjFuncValue();

    return objFuncValue;
}

scalar DAObjFunc::getObjFuncValue()
{
    /*
    Description:
        Call the calcObjFunc in the child class and return
        objFuncValue_
    
        NOTE: This is a interface for external calls where users
        only want compute the objective value based on the existing
        variable fields in OpenFOAM; no need to update state variables
        or point coordinates. This can be used in the primal solver to print
        the objective for each time step.
    */
    // calculate
    this->calcObjFunc(
        objFuncFaceSources_,
        objFuncCellSources_,
        objFuncFaceValues_,
        objFuncCellValues_,
        objFuncValue_);

    // return
    return objFuncValue_;
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
