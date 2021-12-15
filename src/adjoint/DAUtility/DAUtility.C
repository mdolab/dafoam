/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v2

\*---------------------------------------------------------------------------*/

#include "DAUtility.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

// Constructors
DAUtility::DAUtility()
{
}

DAUtility::~DAUtility()
{
}

void DAUtility::pyDict2OFDict(
    PyObject* pyDict,
    dictionary& ofDict)
{
    /*
    Description:
        Parse a Python dictionary to an OpenFOAM dictionary
        Only support a certain number of data types

    Input:
        pyDict: Pytion dictionary

    Output:
        ofDict: OpenFoam dictionary
    
    Example:
        We support two types of pyDict, one is to set the key value
        as a list and have the value type as the first element.
        This is needed in pyDAFoam.py to reminder users the date type
        of a key value

        pyDict = {
            "solverName": [str, "DASimpleFoam"],
            "patches": [list, [1.0, 2.0]]
        }

        The second type of pyDict is just like a usual dict:

        pyDict = {
            "solverName": "DASimpleFoam",
            "patches": [1.0, 2.0]
        }

        Note: one can have multiple levels of sub dictionaries
    */

    //PyObject_Print(pyDict,stdout,0);Info<<endl;

    // size of the pyOpions keys
    Py_ssize_t dictSize = PyDict_Size(pyDict);
    // all the keys
    PyObject* keys = PyDict_Keys(pyDict);
    // loop over all the keys in pyDict and assign their values
    // to ofDict
    for (label i = 0; i < dictSize; i++)
    {
        // the ith key
        PyObject* keyI = PyList_GetItem(keys, i);
        // convert it to UTF8 such that we can use it in C++
        const char* keyUTF8 = PyUnicode_AsUTF8(keyI);

        //std::cout << "Key is "<<keyUTF8<<std::endl;
        // the actual value of this key, NOTE: it is a list
        // the 1st one is its type and the 2nd one is its value
        // this is because we need to make sure users prescribe
        // the right data type in pyDAFoam.py, so we set the
        // value like this in pyDAFoam.py:
        // defOptions = {
        // "solverName": [str, "DASimpleFOam"] }
        // however, the above has one exception when using subDict
        // in this case, we only have one element
        PyObject* value = PyDict_GetItem(pyDict, keyI);
        const char* valueTypeTmp = Py_TYPE(value)->tp_name;
        PyObject* value1;
        if (word(valueTypeTmp) == "list")
        {
            // there are two possibilities for this case
            // for nonSubDict case, we need to have pyDict format
            // like this: { "solverName": [str, "DASimpleDAFoam"] }
            // however, for subDicts, we dont specify the type property
            PyObject* value0 = PyList_GetItem(value, 0);
            const char* valueTypeTmp0 = Py_TYPE(value0)->tp_name;
            if (word(valueTypeTmp0) == "type")
            {
                // this is nonSubdict case
                // the second element of value is the actual value
                value1 = PyList_GetItem(value, 1);
            }
            else
            {
                // it is for subdicts, so value1=value
                value1 = value;
            }
        }
        else
        {
            // if we only have one element, set value1 = value
            value1 = value;
        }

        //PyObject_Print(value1,stdout,0);Info<<endl;
        // get the type of value1
        const char* valueType = Py_TYPE(value1)->tp_name;
        if (word(valueType) == "str")
        {
            const char* valSet = PyUnicode_AsUTF8(value1);
            ofDict.add(keyUTF8, word(valSet));
        }
        else if (word(valueType) == "int")
        {
            long valSet = PyLong_AsLong(value1);
            ofDict.add(keyUTF8, label(valSet));
        }
        else if (word(valueType) == "bool")
        {
            label valSet = PyObject_IsTrue(value1);
            ofDict.add(keyUTF8, valSet);
        }
        else if (word(valueType) == "float")
        {
            scalar valSet = PyFloat_AS_DOUBLE(value1);
            ofDict.add(keyUTF8, valSet);
        }
        else if (word(valueType) == "list")
        {
            // size of the list
            Py_ssize_t listSize = PyList_Size(value1);

            //Info<<listSize<<endl;
            // create OpenFOAM lists to hold this list
            // we create all the possible types
            scalarList valSetScalar;
            valSetScalar.setSize(label(listSize));
            labelList valSetLabel;
            valSetLabel.setSize(label(listSize));
            List<word> valSetWord;
            valSetWord.setSize(label(listSize));

            // need to check what type of list this is
            // by checking its first element
            PyObject* tmp = PyList_GetItem(value1, 0);
            const char* tmpType = Py_TYPE(tmp)->tp_name;
            word tmpTypeWord = word(tmpType);
            // assign value to the OpenFOAM list
            for (label j = 0; j < listSize; j++)
            {
                PyObject* valueListJ = PyList_GetItem(value1, j);
                if (tmpTypeWord == "str")
                {
                    const char* valSet = PyUnicode_AsUTF8(valueListJ);
                    valSetWord[j] = word(valSet);
                }
                else if (tmpTypeWord == "int")
                {
                    long valSet = PyLong_AsLong(valueListJ);
                    valSetLabel[j] = label(valSet);
                }
                else if (tmpTypeWord == "float")
                {
                    scalar valSet = PyFloat_AS_DOUBLE(valueListJ);
                    valSetScalar[j] = valSet;
                }
                else if (tmpTypeWord == "bool")
                {
                    label valSet = PyObject_IsTrue(valueListJ);
                    valSetLabel[j] = valSet;
                }
                else
                {
                    FatalErrorIn("pyDict2OFDict") << "Type: <" << tmpTypeWord << "> for " << keyUTF8
                                                  << " list is not supported! Options are:"
                                                  << " str, int, bool, and float!"
                                                  << abort(FatalError);
                }
            }

            // add the list to the ofDict dict
            if (tmpTypeWord == "str")
            {
                ofDict.add(keyUTF8, valSetWord);
            }
            else if (tmpTypeWord == "int")
            {
                ofDict.add(keyUTF8, valSetLabel);
            }
            else if (tmpTypeWord == "float")
            {
                ofDict.add(keyUTF8, valSetScalar);
            }
            else if (tmpTypeWord == "bool")
            {
                ofDict.add(keyUTF8, valSetLabel);
            }
        }
        else if (word(valueType) == "dict")
        {
            // if its a subdict, recursely call this function
            dictionary subDict;
            DAUtility::pyDict2OFDict(value1, subDict);
            ofDict.add(keyUTF8, subDict);
        }
        else
        {
            FatalErrorIn("pyDict2OFDict") << "Type: " << valueType << " for " << keyUTF8
                                          << " is not supported! Options are: str, int, float, bool, list, and dict!"
                                          << abort(FatalError);
        }
        //std::cout << "My type is " << valueType << std::endl;
    }
}

void DAUtility::readVectorBinary(
    Vec vecIn,
    const word prefix)
{
    /*
    Description:
        Read a vector in binary form

    Input:
        vecIn: a Petsc vector to read values into (also output)
        prefix: Name of the Petsc vector from disk

    Example:
        If the vector storing in the disk reads: dFdWVector.bin
        Then read the vector using:
        Vec dFdW;
        codes to initialize vector dFdW....
        readVectorBinary(dFdW,"dFdwVector");
        NOTE: the prefix does not include ".bin"
    */

    std::ostringstream fileNameStream("");
    fileNameStream << prefix << ".bin";
    word fileName = fileNameStream.str();

    PetscViewer viewer;
    PetscViewerBinaryOpen(PETSC_COMM_WORLD, fileName.c_str(), FILE_MODE_READ, &viewer);
    VecLoad(vecIn, viewer);
    PetscViewerDestroy(&viewer);

    return;
}

void DAUtility::writeVectorBinary(
    const Vec vecIn,
    const word prefix)
{
    /*
    Description:
        Write a vector in binary form

    Input:
        vecIn: a Petsc vector to write to disk
        prefix: Name of the Petsc vector to write to disk

    Example:
        Vec dFdW;
        codes to initialize vector dFdW....
        writeVectorBinary(dFdW,"dFdwVector");
        This will write the dFdW vector to disk with name "dFdWVector.bin"
        NOTE: the prefix does not include ".bin"
    */

    std::ostringstream fileNameStream("");
    fileNameStream << prefix << ".bin";
    word fileName = fileNameStream.str();

    PetscViewer viewer;
    PetscViewerBinaryOpen(PETSC_COMM_WORLD, fileName.c_str(), FILE_MODE_WRITE, &viewer);
    VecView(vecIn, viewer);
    PetscViewerDestroy(&viewer);

    return;
}

void DAUtility::writeVectorASCII(
    const Vec vecIn,
    const word prefix)
{
    /*
    Description:
        Write a vector in ASCII form
    Input:
        vecIn: a Petsc vector to write to disk
        prefix: Name of the Petsc vector to write to disk
    Example:
        Vec dFdW;
        codes to initialize vector dFdW....
        writeVectorASCII(dFdW,"dFdwVector");
        This will write the dFdW vector to disk with name "dFdWVector.dat"
        NOTE: the prefix does not include ".dat"
    */

    std::ostringstream fileNameStream("");
    fileNameStream << prefix << ".dat";
    word fileName = fileNameStream.str();

    PetscViewer viewer;
    PetscViewerASCIIOpen(PETSC_COMM_WORLD, fileName.c_str(), &viewer);
    PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_MATLAB); // write all the digits
    VecView(vecIn, viewer);
    PetscViewerDestroy(&viewer);

    return;
}

void DAUtility::readMatrixBinary(
    Mat matIn,
    const word prefix)
{
    /*
    Description:
        Read a matrix in binary form

    Input:
        matIn: a Petsc matrix to read values into (also output)
        prefix: Name of the Petsc matrix from disk

    Example:
        If the matrix storing in the disk reads: dRdWMat.bin
        Then read the matrix using:
        Mat dRdW;
        codes to initialize matrix dRdW....
        readMatrixBinary(dRdW,"dRdwVector");
        NOTE: the prefix does not include ".bin"
    */

    std::ostringstream fileNameStream("");
    fileNameStream << prefix << ".bin";
    word fileName = fileNameStream.str();

    PetscViewer viewer;
    PetscViewerBinaryOpen(PETSC_COMM_WORLD, fileName.c_str(), FILE_MODE_READ, &viewer);
    MatLoad(matIn, viewer);
    PetscViewerDestroy(&viewer);

    return;
}

void DAUtility::writeMatrixBinary(
    const Mat matIn,
    const word prefix)
{
    /*
    Description:
        Write a matrix in binary form

    Input:
        matIn: a Petsc matrix to write to disk
        prefix: Name of the Petsc matrix to write to disk

    Example:
        Mat dRdW;
        codes to initialize matrix dRdW....
        writeMatrixBinary(dRdW,"dRdWMat");
        This will write the dRdW matrix to disk with name "dRdWMat.bin"
        NOTE: the prefix does not include ".bin"
    */

    std::ostringstream fileNameStream("");
    fileNameStream << prefix << ".bin";
    word fileName = fileNameStream.str();

    PetscViewer viewer;
    PetscViewerBinaryOpen(PETSC_COMM_WORLD, fileName.c_str(), FILE_MODE_WRITE, &viewer);
    MatView(matIn, viewer);
    PetscViewerDestroy(&viewer);

    return;
}

void DAUtility::writeMatrixASCII(
    const Mat matIn,
    const word prefix)
{
    /*
    Description:
        Write a matrix in ASCII form

    Input:
        matIn: a Petsc matrix to write to disk
        prefix: Name of the Petsc matrix to write to disk

    Example:
        Mat dRdW;
        codes to initialize matrix dRdW....
        writeMatrixASCII(dRdW,"dRdWMat");
        This will write the dRdW matrix to disk with name "dRdWMat.dat"
        NOTE: the prefix does not include ".dat"
    */

    std::ostringstream fileNameStream("");
    fileNameStream << prefix << ".dat";
    word fileName = fileNameStream.str();

    PetscViewer viewer;
    PetscViewerASCIIOpen(PETSC_COMM_WORLD, fileName.c_str(), &viewer);
    MatView(matIn, viewer);
    PetscViewerDestroy(&viewer);

    return;
}

void DAUtility::boundVar(
    const dictionary& allOptions,
    volScalarField& var,
    const label printToScreen)
{
    /*
    Description:
        Bound a field variable according to the bounds defined
        in the allOptions dict. This is a overload function for volScalarField

    Input:
        allOptions: a dictionary that has upper and lower bound values
        We need to give specific name for the bounds, i.e.,
        
        variable name + Max   -> setting upper bound

        variable name + Min   -> setting lower bound
    
    Input & Output:
        var: an OpenFOAM field variable to be bounded

    Example:
        dictionary allOptions;
        allOptions.set("pMax", 120000.0);
        allOptions.set("pMin", 80000.0);
        DAUtility daUtil;
        volScalarField p = .... // initialize p
        daUtil.boundVar(allOptions, p);
    */

    const scalar vGreat = 1e200;
    label useUpperBound = 0, useLowerBound = 0;

    dictionary varBoundsDict = allOptions.subDict("primalVarBounds");

    word lowerBoundName = var.name() + "Min";
    word upperBoundName = var.name() + "Max";

    scalar varMin = varBoundsDict.lookupOrDefault<scalar>(lowerBoundName, -vGreat);
    scalar varMax = varBoundsDict.lookupOrDefault<scalar>(upperBoundName, vGreat);

    forAll(var, cellI)
    {
        if (var[cellI] <= varMin)
        {
            var[cellI] = varMin;
            useLowerBound = 1;
        }
        if (var[cellI] >= varMax)
        {
            var[cellI] = varMax;
            useUpperBound = 1;
        }
    }

    forAll(var.boundaryField(), patchI)
    {
        forAll(var.boundaryField()[patchI], faceI)
        {
            if (var.boundaryFieldRef()[patchI][faceI] <= varMin)
            {
                var.boundaryFieldRef()[patchI][faceI] = varMin;
                useLowerBound = 1;
            }
            if (var.boundaryFieldRef()[patchI][faceI] >= varMax)
            {
                var.boundaryFieldRef()[patchI][faceI] = varMax;
                useUpperBound = 1;
            }
        }
    }

    if (printToScreen)
    {
        if (useUpperBound)
        {
            Info << "Bounding " << var.name() << "<" << varMax << endl;
        }
        if (useLowerBound)
        {
            Info << "Bounding " << var.name() << ">" << varMin << endl;
        }
    }

    return;
}

void DAUtility::boundVar(
    const dictionary& allOptions,
    volVectorField& var,
    const label printToScreen)
{
    /*
    Description:
        Bound a field variable according to the bounds defined
        in the allOptions dict. This is a overload function for volVectorField

    Input:
        allOptions: a dictionary that has upper and lower bound values
        We need to give specific name for the bounds, i.e.,
        
        variable name + Max   -> setting upper bound

        variable name + Min   -> setting lower bound
    
    Input & Output:
        var: an OpenFOAM field variable to be bounded

    Example:
        dictionary allOptions;
        allOptions.set("pMax", 120000.0);
        allOptions.set("pMin", 80000.0);
        DAUtility daUtil;
        volScalarField p = .... // initialize p
        daUtil.boundVar(allOptions, p);
    */

    const scalar vGreat = 1e200;
    label useUpperBound = 0, useLowerBound = 0;

    dictionary varBoundsDict = allOptions.subDict("primalVarBounds");

    word lowerBoundName = var.name() + "Min";
    word upperBoundName = var.name() + "Max";

    scalar varMin = varBoundsDict.lookupOrDefault<scalar>(lowerBoundName, -vGreat);
    scalar varMax = varBoundsDict.lookupOrDefault<scalar>(upperBoundName, vGreat);

    forAll(var, cellI)
    {
        for (label i = 0; i < 3; i++)
        {
            if (var[cellI][i] <= varMin)
            {
                var[cellI][i] = varMin;
                useLowerBound = 1;
            }
            if (var[cellI][i] >= varMax)
            {
                var[cellI][i] = varMax;
                useUpperBound = 1;
            }
        }
    }

    forAll(var.boundaryField(), patchI)
    {
        forAll(var.boundaryField()[patchI], faceI)
        {
            for (label i = 0; i < 3; i++)
            {
                if (var.boundaryFieldRef()[patchI][faceI][i] <= varMin)
                {
                    var.boundaryFieldRef()[patchI][faceI][i] = varMin;
                    useLowerBound = 1;
                }
                if (var.boundaryFieldRef()[patchI][faceI][i] >= varMax)
                {
                    var.boundaryFieldRef()[patchI][faceI][i] = varMax;
                    useUpperBound = 1;
                }
            }
        }
    }

    if (printToScreen)
    {
        if (useUpperBound)
        {
            Info << "Bounding " << var.name() << "<" << varMax << endl;
        }
        if (useLowerBound)
        {
            Info << "Bounding " << var.name() << ">" << varMin << endl;
        }
    }

    return;
}

globalIndex DAUtility::genGlobalIndex(const label localIndexSize)
{
    /*
    Generate a glocal index system based on the local index size 
    such that we can use it to map a local index to a global one

    Input:
    -----
    localIndexSize: the SIZE of local index

    Output:
    ------
    globalIndex object: the global index object to map a local index
    to a global index

    Example:
    --------
    If the local index reads:

    On processor 0:
    labelList sampleList = {0, 1, 2};
    globalIndex glbSample = genGlobalIndex(sampleList.size());

    On processor 1:
    labelList sampleList = {0, 1};
    globalIndex glbSample = genGlobalIndex(sampleList.size());

    After each processor calls genGlobalIndex and get the glbSample
    object, we can use it to map a local index to a global one,
    e.g., on processor 0, if we call:

    label glxIdx = glbSample.toGlobal(1);

    it will return glbIdx = 1;
    However, on processor 1, if we call

    label glxIdx = glbSample.toGlobal(1);

    it will return glbIdx = 4;

    The date storage structure is illustrated as follows

    global index -> 0 1 2 3 4
    local index  -> 0 1 2 0 1
                    ----- ===
                    proc0 proc1

    */
    globalIndex result(localIndexSize);
    return result;
}

label DAUtility::isValueCloseToRef(
    const scalar val,
    const scalar refVal,
    const scalar tol)
{
    /* 
    Description:
        check whether a value is close to a reference value by a tolerance

    Input:
        val: value to check
        refVal: reference value 
        tol: tolerance
    
    Output:
        return: 1 means fabs(val-refVal) < tol
    */
    if (fabs(val - refVal) < tol)
    {
        return 1;
    }
    else
    {
        return 0;
    }
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
