/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v2

\*---------------------------------------------------------------------------*/
#include "TestDAFoamIncompressible.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

// Constructors
TestDAFoamIncompressible::TestDAFoamIncompressible(char* argsAll)
    : argsAll_(argsAll)
{
}

TestDAFoamIncompressible::~TestDAFoamIncompressible()
{
}

label TestDAFoamIncompressible::validateOFDict(const dictionary& ofDict)
{
    // This is how ofDict should look like, it contains
    // all the supported types
    /*
    {
        key1            15;
        key2            5.5;
        key3            solver1;
        key4            0;
        key5            3 ( 1 2 3 );
        key6            3 ( 1.5 2.3 3.4 );
        key7            3 ( ele1 ele2 ele3 );
        key8            3 ( 0 1 0 );
        key9
        {
            subkey1         30;
            subkey2         3.5;
            subkey3         solver2;
            subkey4         1;
            subkey5         3 ( 4 5 6 );
            subkey6         3 ( 2.5 7.7 8.9 );
            subkey7         3 ( ele4 ele5 ele6 );
            subkey8         3 ( 1 0 1 );
        }
    }
    */

    label testErrors = 0;
    // Now check if ofDict is properly set
    if (readLabel(ofDict.lookup("key1")) != 15)
    {
        Pout << "Error: key1 error in pyDict2OFDict!" << endl;
        testErrors += 1;
    }

    if (readScalar(ofDict.lookup("key2")) != 5.5)
    {
        Pout << "Error: key2 error in pyDict2OFDict!" << endl;
        testErrors += 1;
    }

    if (word(ofDict.lookup("key3")) != "solver1")
    {
        Pout << "Error: key3 error in pyDict2OFDict!" << endl;
        testErrors += 1;
    }

    if (readLabel(ofDict.lookup("key4")) != 0)
    {
        Pout << "Error: key4 error in pyDict2OFDict!" << endl;
        testErrors += 1;
    }

    labelList list1;
    ofDict.readEntry("key5", list1);
    if (list1[0] != 1 || list1[1] != 2 || list1[2] != 3)
    {
        Pout << "Error: key5 error in pyDict2OFDict!" << endl;
        testErrors += 1;
    }

    scalarList list2;
    ofDict.readEntry("key6", list2);
    if (list2[0] != 1.5 || list2[1] != 2.3 || list2[2] != 3.4)
    {
        Pout << "Error: key6 error in pyDict2OFDict!" << endl;
        testErrors += 1;
    }

    wordList list3;
    ofDict.readEntry("key7", list3);
    if (list3[0] != "ele1" || list3[1] != "ele2" || list3[2] != "ele3")
    {
        Pout << "Error: key7 error in pyDict2OFDict!" << endl;
        testErrors += 1;
    }

    labelList list4;
    ofDict.readEntry("key8", list4);
    if (list4[0] != 0 || list4[1] != 1 || list4[2] != 0)
    {
        Pout << "Error: key8 error in pyDict2OFDict!" << endl;
        testErrors += 1;
    }

    // check subDict
    dictionary subDict1;
    subDict1 = ofDict.subDict("key9");

    if (readLabel(subDict1.lookup("subkey1")) != 30)
    {
        Pout << "Error: subkey1 error in pyDict2OFDict!" << endl;
        testErrors += 1;
    }

    if (readScalar(subDict1.lookup("subkey2")) != 3.5)
    {
        Pout << "Error: subkey2 error in pyDict2OFDict!" << endl;
        testErrors += 1;
    }

    if (word(subDict1.lookup("subkey3")) != "solver2")
    {
        Pout << "Error: subkey3 error in pyDict2OFDict!" << endl;
        testErrors += 1;
    }

    if (readLabel(subDict1.lookup("subkey4")) != 1)
    {
        Pout << "Error: subkey4 error in pyDict2OFDict!" << endl;
        testErrors += 1;
    }

    labelList subList1;
    subDict1.readEntry("subkey5", subList1);
    if (subList1[0] != 4 || subList1[1] != 5 || subList1[2] != 6)
    {
        Pout << "Error: subkey5 error in pyDict2OFDict!" << endl;
        testErrors += 1;
    }

    scalarList subList2;
    subDict1.readEntry("subkey6", subList2);
    if (subList2[0] != 2.5 || subList2[1] != 7.7 || subList2[2] != 8.9)
    {
        Pout << "Error: subkey6 error in pyDict2OFDict!" << endl;
        testErrors += 1;
    }

    wordList subList3;
    subDict1.readEntry("subkey7", subList3);
    if (subList3[0] != "ele4" || subList3[1] != "ele5" || subList3[2] != "ele6")
    {
        Pout << "Error: subkey7 error in pyDict2OFDict!" << endl;
        testErrors += 1;
    }

    labelList subList4;
    subDict1.readEntry("subkey8", subList4);
    if (subList4[0] != 1 || subList4[1] != 0 || subList4[2] != 1)
    {
        Pout << "Error: key8 error in pyDict2OFDict!" << endl;
        testErrors += 1;
    }

    if (testErrors != 0)
    {
        Pout << "Error: ofDict" << ofDict << endl;
    }

    return testErrors;
}

label TestDAFoamIncompressible::testDAUtility(
    PyObject* pyDict,
    PyObject* pyStrList)
{
    autoPtr<argList> argsPtr_;
#include "setArgs.H"
#include "setRootCasePython.H"
#include "createTime.H"
#include "createMesh.H"

    label testErrors = 0;

    // ********************* pyDict2OFDict *********************
    dictionary ofDict;
    DAUtility::pyDict2OFDict(pyDict, ofDict);

    testErrors = validateOFDict(ofDict);

    // test DAUtility::pyStrList2OFWordList
    //  ofWordListRef = {"a", "B", "C1"};
    wordList ofWordList;
    DAUtility::pyStrList2OFWordList(pyStrList, ofWordList);
    if (ofWordList[0] != "a" || ofWordList[1] != "B" || ofWordList[2] != "C1")
    {
        Pout << ofWordList << endl;
        Pout << "Error: ofWordList error in pyStrList2OFWordList!" << endl;
        testErrors += 1;
    }

    // ********************* List functions *********************
    // DAUtility::isInList
    scalarList sList = {1.0, 2.3, 3.5};
    labelList lList = {1, 2, 3};
    wordList wList = {"e1", "e2", "e3"};
    if (!DAUtility::isInList<scalar>(1.0, sList) || DAUtility::isInList<scalar>(5.0, sList))
    {
        Pout << "ERROR: scalar error in isInList!" << endl;
        testErrors += 1;
    }
    if (!DAUtility::isInList<label>(1, lList) || DAUtility::isInList<label>(4, lList))
    {
        Pout << "ERROR: label error in isInList!" << endl;
        testErrors += 1;
    }
    if (!DAUtility::isInList<word>("e1", wList) || DAUtility::isInList<word>("e5", wList))
    {
        Pout << "ERROR: word error in isInList!" << endl;
        testErrors += 1;
    }

    // DAUtility::listReplaceVal
    scalarList sListRe = {1.0, 2.3, 3.5};
    labelList lListRe = {1, 2, 3};
    wordList wListRe = {"e1", "e2", "e3"};

    scalarList sListReNew = {5.5, 2.3, 3.5};
    labelList lListReNew = {1, 5, 3};
    wordList wListReNew = {"e1", "e2", "e10"};

    DAUtility::listReplaceVal<scalar>(sListRe, 1.0, 5.5);
    DAUtility::listReplaceVal<label>(lListRe, 2, 5);
    DAUtility::listReplaceVal<word>(wListRe, "e3", "e10");

    if (sListRe != sListReNew)
    {
        Pout << "ERROR: scalar error in listReplaceVal!" << endl;
        testErrors += 1;
    }
    if (lListRe != lListReNew)
    {
        Pout << "ERROR: label error in listReplaceVal!" << endl;
        testErrors += 1;
    }
    if (wListRe != wListReNew)
    {
        Pout << "ERROR: word error in listReplaceVal!" << endl;
        testErrors += 1;
    }

    // DAUtility::listDeleteVal
    scalarList sListDel = {1.0, 2.3, 3.5, 1.0}; // delete multiple
    labelList lListDel = {1, 2, 3};
    wordList wListDel = {"e1", "e2", "e3"};

    scalarList sListDelNew = {2.3, 3.5};
    labelList lListDelNew = {1, 3};
    wordList wListDelNew = {"e1", "e2"};

    DAUtility::listDeleteVal<scalar>(sListDel, 1.0);
    DAUtility::listDeleteVal<label>(lListDel, 2);
    DAUtility::listDeleteVal<word>(wListDel, "e3");

    if (sListDel != sListDelNew)
    {
        Pout << "ERROR: scalar error in listDeleteVal!" << endl;
        testErrors += 1;
    }
    if (lListDel != lListDelNew)
    {
        Pout << "ERROR: label error in listDeleteVal!" << endl;
        testErrors += 1;
    }
    if (wListDel != wListDelNew)
    {
        Pout << "ERROR: word error in listDeleteVal!" << endl;
        testErrors += 1;
    }

    // ********************* MAT VEC IO *********************
    Vec tVec;
    label myProcNo = Pstream::myProcNo();
    VecCreate(PETSC_COMM_WORLD, &tVec);
    VecSetSizes(tVec, myProcNo + 1, PETSC_DETERMINE);
    VecSetFromOptions(tVec);
    VecZeroEntries(tVec);

    PetscScalar* tVecArray;
    VecGetArray(tVec, &tVecArray);
    for (label i = 0; i < myProcNo + 1; i++)
    {
        tVecArray[i] = i * 2.0;
    }

    VecRestoreArray(tVec, &tVecArray);

    VecAssemblyBegin(tVec);
    VecAssemblyEnd(tVec);

    DAUtility::writeVectorBinary(tVec, "tVec");
    DAUtility::writeVectorASCII(tVec, "tVec");

    Vec tVecRead;
    VecCreate(PETSC_COMM_WORLD, &tVecRead);
    VecSetSizes(tVecRead, myProcNo + 1, PETSC_DETERMINE);
    VecSetFromOptions(tVecRead);
    VecZeroEntries(tVecRead);
    DAUtility::readVectorBinary(tVecRead, "tVec");

    PetscBool equalVec;
    VecEqual(tVec, tVecRead, &equalVec);
    if (!equalVec)
    {
        Pout << "ERROR: error in read/writeVectorBinary!" << endl;
        testErrors += 1;
    }

    Mat tMat;
    MatCreate(PETSC_COMM_WORLD, &tMat);
    MatSetSizes(tMat, myProcNo + 1, PETSC_DECIDE, PETSC_DETERMINE, 2);
    MatSetFromOptions(tMat);
    MatMPIAIJSetPreallocation(tMat, 2, NULL, 2, NULL);
    MatSeqAIJSetPreallocation(tMat, 2, NULL);
    MatSetUp(tMat);

    PetscInt Istart, Iend;
    MatGetOwnershipRange(tMat, &Istart, &Iend);
    for (PetscInt i = Istart; i < Iend; i++)
    {
        for (label j = 0; j < 2; j++)
        {
            scalar val = i * 2.0 + j;
            MatSetValue(tMat, i, j, val, INSERT_VALUES);
        }
    }
    MatAssemblyBegin(tMat, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(tMat, MAT_FINAL_ASSEMBLY);

    DAUtility::writeMatrixBinary(tMat, "tMat");
    DAUtility::writeMatrixASCII(tMat, "tMat");

    Mat tMatRead;
    MatCreate(PETSC_COMM_WORLD, &tMatRead);
    MatSetSizes(tMatRead, myProcNo + 1, PETSC_DECIDE, PETSC_DETERMINE, 2);
    MatSetFromOptions(tMatRead);
    MatMPIAIJSetPreallocation(tMatRead, 2, NULL, 2, NULL);
    MatSeqAIJSetPreallocation(tMatRead, 2, NULL);
    MatSetUp(tMatRead);
    DAUtility::readMatrixBinary(tMatRead, "tMat");

    PetscBool equalMat;
    MatEqual(tMat, tMatRead, &equalMat);
    if (!equalMat)
    {
        Pout << "ERROR: error in read/writeMatrixBinary!" << endl;
        testErrors += 1;
    }

    // test DAUtility::boundVar
    volScalarField T(
        IOobject(
            "T1",
            runTime.timeName(),
            runTime,
            IOobject::NO_READ,
            IOobject::NO_WRITE),
        mesh,
        dimensionedScalar("T1", dimensionSet(0, 0, 0, 1, 0, 0, 0), 1.0),
        zeroGradientFvPatchScalarField::typeName);

    forAll(T, idxI)
    {
        T[idxI] = 1.0 * Foam::sin(idxI * 1.0);
    }
    forAll(T.boundaryField(), patchI)
    {
        forAll(T.boundaryField()[patchI], faceI)
        {
            T.boundaryFieldRef()[patchI][faceI] = 1.0 * Foam::sin(patchI * faceI * 1.0);
        }
    }
    T.write();

    volVectorField U(
        IOobject(
            "U1",
            runTime.timeName(),
            runTime,
            IOobject::NO_READ,
            IOobject::NO_WRITE),
        mesh,
        dimensionedVector("U1", dimensionSet(0, 1, -1, 0, 0, 0, 0), vector::zero),
        zeroGradientFvPatchVectorField::typeName);
    forAll(U, idxI)
    {
        for (label i = 0; i < 3; i++)
        {
            U[idxI][i] = 1.0 * Foam::sin(idxI * i * 1.0);
        }
    }
    forAll(U.boundaryField(), patchI)
    {
        forAll(U.boundaryField()[patchI], faceI)
        {
            for (label i = 0; i < 3; i++)
            {
                U.boundaryFieldRef()[patchI][faceI][i] = 1.0 * Foam::sin(patchI * faceI * i * 1.0);
            }
        }
    }
    U.write();

    dictionary boundVarSubDict;
    boundVarSubDict.set("T1UpperBound", 0.5);
    boundVarSubDict.set("T1LowerBound", -0.2);
    boundVarSubDict.set("U1UpperBound", 0.6);
    boundVarSubDict.set("U1LowerBound", -0.3);
    dictionary boundVarDict;
    boundVarDict.set("primalVarBounds", boundVarSubDict);
    Info << boundVarDict << endl;
    DAUtility::boundVar(boundVarDict, U);
    DAUtility::boundVar(boundVarDict, T);

    forAll(U, idxI)
    {
        for (label i = 0; i < 3; i++)
        {
            if (U[idxI][i] > 0.6 || U[idxI][i] < -0.3)
            {
                Pout << "ERROR: error in boundVar vector!" << endl;
                testErrors += 1;
            }
        }
    }
    forAll(U.boundaryField(), patchI)
    {
        forAll(U.boundaryField()[patchI], faceI)
        {
            for (label i = 0; i < 3; i++)
            {
                if (U.boundaryField()[patchI][faceI][i] > 0.6 || U.boundaryField()[patchI][faceI][i] < -0.3)
                {
                    Pout << "ERROR: error in boundVar vector!" << endl;
                    testErrors += 1;
                }
            }
        }
    }

    forAll(T, idxI)
    {
        if (T[idxI] > 0.5 || T[idxI] < -0.2)
        {
            Pout << "ERROR: error in boundVar scalar!" << endl;
            testErrors += 1;
        }
    }
    forAll(T.boundaryField(), patchI)
    {
        forAll(T.boundaryField()[patchI], faceI)
        {
            if (T.boundaryField()[patchI][faceI] > 0.5 || T.boundaryField()[patchI][faceI] < -0.2)
            {
                Pout << "ERROR: error in boundVar scalar!" << endl;
                testErrors += 1;
            }
        }
    }

    T.rename("T1_bound");
    T.write();
    U.rename("U1_bound");
    U.write();

    // test DAUtility::genGlobalIndex
    label nProcs = Pstream::nProcs();
    globalIndex testGlbIdx;
    testGlbIdx = DAUtility::genGlobalIndex(2);
    label glbI = testGlbIdx.toGlobal(0);
    reduce(glbI, sumOp<label>());
    label refGlbI = nProcs * (nProcs - 1);
    if (glbI != refGlbI)
    {
        Pout << "ERROR: error genGlobalIndex!" << endl;
        testErrors += 1;
    }

    // test DAUtility::isValueCloseToRef
    scalar valA1 = 1.00000000000001;
    scalar valA2 = 0.99999999999999;
    scalar valA3 = 1.0001;
    scalar valARef = 1.0;
    if (!DAUtility::isValueCloseToRef(valA1, valARef)
        || !DAUtility::isValueCloseToRef(valA2, valARef)
        || DAUtility::isValueCloseToRef(valA3, valARef))
    {
        Pout << "ERROR: error isValueCloseToRef!" << endl;
        testErrors += 1;
    }

    return testErrors;
}

label TestDAFoamIncompressible::testDAOption(PyObject* pyDict)
{
    autoPtr<argList> argsPtr_;
#include "setArgs.H"
#include "setRootCasePython.H"
#include "createTime.H"
#include "createMesh.H"

    label testErrors = 0;

    DAOption daOption(mesh, pyDict);

    // test getAllOptions
    const dictionary& allOptions = daOption.getAllOptions();
    testErrors = validateOFDict(allOptions);

    // test getOption
    if (daOption.getOption<label>("key1") != 15)
    {
        Pout << "ERROR: key1 error in getOption!" << endl;
        testErrors += 1;
    }

    if (daOption.getOption<scalar>("key2") != 5.5)
    {
        Pout << "ERROR: key2 error in getOption!" << endl;
        testErrors += 1;
    }

    if (daOption.getOption<word>("key3") != "solver1")
    {
        Pout << "ERROR: key3 error in getOption!" << endl;
        testErrors += 1;
    }

    labelList list5 = daOption.getOption<labelList>("key5");
    labelList list5Ref = {1, 2, 3};
    if (list5 != list5Ref)
    {
        Pout << "ERROR: key5 error in getOption!" << endl;
        testErrors += 1;
    }

    scalarList list6 = daOption.getOption<scalarList>("key6");
    scalarList list6Ref = {1.5, 2.3, 3.4};
    if (list6 != list6Ref)
    {
        Pout << "ERROR: key6 error in getOption!" << endl;
        testErrors += 1;
    }

    wordList list7 = daOption.getOption<wordList>("key7");
    wordList list7Ref = {"ele1", "ele2", "ele3"};
    if (list7 != list7Ref)
    {
        Pout << "ERROR: key7 error in getOption!" << endl;
        testErrors += 1;
    }

    // test getSubDictOption
    if (daOption.getSubDictOption<label>("key9", "subkey1") != 30)
    {
        Pout << "ERROR: key9 subkey1 error in getOption!" << endl;
        testErrors += 1;
    }

    if (daOption.getSubDictOption<scalar>("key9", "subkey2") != 3.5)
    {
        Pout << "ERROR: key9 subkey2 error in getOption!" << endl;
        testErrors += 1;
    }

    if (daOption.getSubDictOption<word>("key9", "subkey3") != "solver2")
    {
        Pout << "ERROR: key9 subkey3 error in getOption!" << endl;
        testErrors += 1;
    }

    labelList subList5 = daOption.getSubDictOption<labelList>("key9", "subkey5");
    labelList subList5Ref = {4, 5, 6};
    if (subList5 != subList5Ref)
    {
        Pout << "ERROR: key9 subkey5 error in getOption!" << endl;
        testErrors += 1;
    }

    scalarList subList6 = daOption.getSubDictOption<scalarList>("key9", "subkey6");
    scalarList subList6Ref = {2.5, 7.7, 8.9};
    if (subList6 != subList6Ref)
    {
        Pout << "ERROR: key9 subkey6 error in getOption!" << endl;
        testErrors += 1;
    }

    wordList subList7 = daOption.getSubDictOption<wordList>("key9", "subkey7");
    wordList subList7Ref = {"ele4", "ele5", "ele6"};
    if (subList7 != subList7Ref)
    {
        Pout << "ERROR: key9 subkey7 error in getOption!" << endl;
        testErrors += 1;
    }

    // test setOption
    label newValLabel = 101;
    daOption.setOption<label>("key1", newValLabel);
    label newValLabelCheck = daOption.getOption<label>("key1");
    if (newValLabel != newValLabelCheck)
    {
        Pout << "ERROR: key1 error in setOption!" << endl;
        testErrors += 1;
    }

    scalar newValScalar = 101.5;
    daOption.setOption<scalar>("key2", newValScalar);
    scalar newValScalarCheck = daOption.getOption<scalar>("key2");
    if (newValScalar != newValScalarCheck)
    {
        Pout << "ERROR: key2 error in setOption!" << endl;
        testErrors += 1;
    }

    word newValWord = "solverN";
    daOption.setOption<word>("key3", newValWord);
    word newValWordCheck = daOption.getOption<word>("key3");
    if (newValWord != newValWordCheck)
    {
        Pout << "ERROR: key3 error in setOption!" << endl;
        testErrors += 1;
    }

    labelList labelListNew = {8, 9}; // NOTE we set a different size
    daOption.setOption<labelList>("key5", labelListNew);
    labelList labelListNewCheck = daOption.getOption<labelList>("key5");
    if (labelListNew != labelListNewCheck)
    {
        Pout << "ERROR: key5 error in setOption!" << endl;
        testErrors += 1;
    }

    scalarList scalarListNew = {8.5, 9.2, 10.7};
    daOption.setOption<scalarList>("key6", scalarListNew);
    scalarList scalarListNewCheck = daOption.getOption<scalarList>("key6");
    if (scalarListNew != scalarListNewCheck)
    {
        Pout << "ERROR: key6 error in setOption!" << endl;
        testErrors += 1;
    }

    wordList wordListNew = {"e9", "e5", "e4", "e3"}; // NOTE we set a different size
    daOption.setOption<wordList>("key7", wordListNew);
    wordList wordListNewCheck = daOption.getOption<wordList>("key7");
    if (wordListNew != wordListNewCheck)
    {
        Pout << "ERROR: key7 error in setOption!" << endl;
        testErrors += 1;
    }

    // test setSubDictOption we only test two typical scenarios
    label labelSubDictNew = 101;
    daOption.setSubDictOption<label>("key9", "subkey1", labelSubDictNew);
    label labelSubDictNewCheck = daOption.getSubDictOption<label>("key9", "subkey1");
    if (labelSubDictNew != labelSubDictNewCheck)
    {
        Pout << "ERROR: key9 subkey1 error in setSubDictOption!" << endl;
        testErrors += 1;
    }

    scalarList scalarSubListNew = {8.5, 9.2};
    daOption.setSubDictOption<scalarList>("key9", "subkey6", scalarSubListNew);
    scalarList scalarSubListNewCheck = daOption.getSubDictOption<scalarList>("key9", "subkey6");
    if (scalarSubListNew != scalarSubListNewCheck)
    {
        Pout << "ERROR: key9 subkey6 error in setSudDictOption!" << endl;
        testErrors += 1;
    }

    daOption.write();

    return testErrors;
}

label TestDAFoamIncompressible::testDAModel(PyObject* pyDict)
{
    autoPtr<argList> argsPtr_;
#include "setArgs.H"
#include "setRootCasePython.H"
#include "createTime.H"
#include "createMesh.H"
#include "createSimpleControl.H"
//#include "createFields.H"

    label testErrors = 0;
/*
    wordList modelStates = {"nut", "p", "U"};
    wordList modelStatesRef = {"nuTilda", "p", "U"};

    daModel.correctModelStates(modelStates);

    if (modelStates != modelStatesRef)
    {
        Pout << "ERROR: in correctModelStates!" << endl;
        testErrors += 1;
    }
*/
    return testErrors;
}

label TestDAFoamIncompressible::testDAStateInfo(PyObject* pyDict)
{

    autoPtr<argList> argsPtr_;
#include "setArgs.H"
#include "setRootCasePython.H"
#include "createTime.H"
#include "createMesh.H"
#include "createSimpleControl.H"
//#include "createFields.H"

    label testErrors = 0;
/*
    const HashTable<wordList>& stateInfo = daStateInfo->getStateInfo();

    HashTable<wordList> stateInfoRef;

    stateInfoRef.set("volScalarStates", {});
    stateInfoRef.set("volVectorStates", {});
    stateInfoRef.set("modelStates", {});
    stateInfoRef.set("surfaceScalarStates", {});
    stateInfoRef["volScalarStates"].append("p");
    stateInfoRef["modelStates"].append("nut");
    stateInfoRef["volVectorStates"].append("U");
    stateInfoRef["surfaceScalarStates"].append("phi");

    daModel.correctModelStates(stateInfoRef["modelStates"]);

    if (stateInfo != stateInfoRef)
    {
        Pout << "compressible error in DAStateInfo!" << endl;
        testErrors += 1;
    }
*/
    return testErrors;
}

label TestDAFoamIncompressible::testDAObjFunc(PyObject* pyDict)
{
    autoPtr<argList> argsPtr_;
#include "setArgs.H"
#include "setRootCasePython.H"
#include "createTime.H"
#include "createMesh.H"
#include "createSimpleControl.H"
//#include "createFields.H"

    label testErrors = 0;

    return testErrors;
}

label TestDAFoamIncompressible::testDAField(PyObject* pyDict)
{
    autoPtr<argList> argsPtr_;
#include "setArgs.H"
#include "setRootCasePython.H"
#include "createTime.H"
#include "createMesh.H"
#include "createSimpleControl.H"
//#include "createFields.H"

    label testErrors = 0;
/*
    Vec xvVec;
    VecCreate(PETSC_COMM_WORLD, &xvVec);
    VecSetSizes(xvVec, mesh.nPoints() * 3, PETSC_DETERMINE);
    VecSetFromOptions(xvVec);

    daField.ofMesh2PointVec(xvVec);
    scalar vecNormOrig;
    VecNorm(xvVec, NORM_2, &vecNormOrig);
    daField.pointVec2OFMesh(xvVec);
    daField.ofMesh2PointVec(xvVec);
    scalar vecNormNew;
    VecNorm(xvVec, NORM_2, &vecNormNew);
    if (vecNormOrig != vecNormNew)
    {
        Pout << "ERROR: pointVec2OFMesh/ofMesh2PointVec error!" << endl;
        testErrors += 1;
    }

    volVectorField URes(
        IOobject(
            "URes",
            mesh.time().timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE,
            true),
        mesh,
        dimensionedVector("URes", dimensionSet(0, 0, 0, 0, 0, 0, 0), vector::one),
        zeroGradientFvPatchVectorField::typeName);

    volScalarField pRes(
        IOobject(
            "pRes",
            mesh.time().timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE,
            true),
        mesh,
        dimensionedScalar("pRes", dimensionSet(0, 0, 0, 0, 0, 0, 0), 2.0),
        zeroGradientFvPatchScalarField::typeName);

    volScalarField nuTildaRes(
        IOobject(
            "nuTildaRes",
            mesh.time().timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE,
            true),
        mesh,
        dimensionedScalar("nuTildaRes", dimensionSet(0, 0, 0, 0, 0, 0, 0), 3.0),
        zeroGradientFvPatchScalarField::typeName);

    surfaceScalarField phiRes(
        IOobject(
            "phiRes",
            mesh.time().timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE,
            true),
        phi);

    Vec resVec;
    VecCreate(PETSC_COMM_WORLD, &resVec);
    VecSetSizes(resVec, daIndex.nLocalAdjointStates, PETSC_DETERMINE);
    VecSetFromOptions(resVec);

    daField.ofResField2ResVec(resVec);
    scalar resVecNormOrig;
    VecNorm(resVec, NORM_2, &resVecNormOrig);
    daField.resVec2OFResField(resVec);
    daField.ofResField2ResVec(resVec);
    scalar resVecNormNew;
    VecNorm(resVec, NORM_2, &resVecNormNew);
    if (resVecNormOrig != resVecNormNew)
    {
        Pout << "ERROR: resVec2OFResField/ofResField2ResVec error!" << endl;
        testErrors += 1;
    }
*/
    return testErrors;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
