/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v1.0

\*---------------------------------------------------------------------------*/

#include "AdjointIO.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

// Constructors
AdjointIO::AdjointIO
(
    const fvMesh& mesh
)
    :
    mesh_(mesh),
    // read dict from system/adjointDict
    adjointDict_
    (
        IOobject
        (
            "adjointDict",
            mesh_.time().system(),
            mesh_,
            IOobject::MUST_READ,
            IOobject::NO_WRITE,
            false
        )
    ),
    adjointParameters_
    (
        IOobject
        (
            "adjointParameters",
            mesh_.time().system(),
            mesh_,
            IOobject::NO_READ,
            IOobject::NO_WRITE,
            false
        )
    )
    
{
    
    
    // read adjoint options from system/adjointDict or use default values
    dictionary adjointOptions
    (
        adjointDict_.subDict
        (
            "adjointOptions"
        )
    ); 
    // read
    epsDeriv            = readOptionOrDefault<scalar>(adjointOptions,"epsDeriv",1.0e-6);
    epsDerivFFD         = readOptionOrDefault<scalar>(adjointOptions,"epsDerivFFD",1.0e-4);
    epsDerivXv          = readOptionOrDefault<scalar>(adjointOptions,"epsDerivXv",1.0e-7);
    epsDerivUIn         = readOptionOrDefault<scalar>(adjointOptions,"epsDerivUIn",1.0e-5);
    epsDerivVis         = readOptionOrDefault<scalar>(adjointOptions,"epsDerivVis",1.0e-8);
    minTolJac           = readOptionOrDefault<scalar>(adjointOptions,"minTolJac",1.0e-14);
    maxTolJac           = readOptionOrDefault<scalar>(adjointOptions,"maxTolJac",1.0e14);
    minTolPC            = readOptionOrDefault<scalar>(adjointOptions,"minTolPC",1.0e-14);
    maxTolPC            = readOptionOrDefault<scalar>(adjointOptions,"maxTolPC",1.0e14);
    transonicPCOption   = readOptionOrDefault<label>(adjointOptions,"transonicPCOption",0);
    stateResetTol       = readOptionOrDefault<scalar>(adjointOptions,"stateResetTol",1.0e-6);
    tractionBCMaxIter   = readOptionOrDefault<label>(adjointOptions,"tractionBCMaxIter",20);
    solveAdjoint        = readBooleanOptionOrDefault(adjointOptions,"solveAdjoint","true");
    useColoring         = readBooleanOptionOrDefault(adjointOptions,"useColoring","true");
    correctWallDist     = readBooleanOptionOrDefault(adjointOptions,"correctWallDist","true");
    reduceResCon4JacMat = readBooleanOptionOrDefault(adjointOptions,"reduceResCon4JacMat","false");
    delTurbProd4PCMat   = readBooleanOptionOrDefault(adjointOptions,"delTurbProd4PCMat","false");
    calcPCMat           = readBooleanOptionOrDefault(adjointOptions,"calcPCMat","true");
    fastPCMat           = readBooleanOptionOrDefault(adjointOptions,"fastPCMat","false");
    writeMatrices       = readBooleanOptionOrDefault(adjointOptions,"writeMatrices","false");
    adjGMRESCalcEigen   = readBooleanOptionOrDefault(adjointOptions,"adjGMRESCalcEigen","false");
    readMatrices        = readBooleanOptionOrDefault(adjointOptions,"readMatrices","false");
    adjJacMatOrdering   = readOptionOrDefault<word>(adjointOptions,"adjJacMatOrdering","state");
    adjJacMatReOrdering = readOptionOrDefault<word>(adjointOptions,"adjJacMatReOrdering","rcm");
    adjSegregated       = readBooleanOptionOrDefault(adjointOptions,"adjSegregated","false");
    segAdjParameters    = readOptionOrDefault< HashTable<scalar> >(adjointOptions,"segAdjParameters",{});
    adjGMRESRestart     = readOptionOrDefault<label>(adjointOptions,"adjGMRESRestart",200);
    adjGlobalPCIters    = readOptionOrDefault<label>(adjointOptions,"adjGlobalPCIters",0);
    adjLocalPCIters     = readOptionOrDefault<label>(adjointOptions,"adjLocalPCIters",1);
    adjASMOverlap       = readOptionOrDefault<label>(adjointOptions,"adjASMOverlap",1);
    adjPCFillLevel      = readOptionOrDefault<label>(adjointOptions,"adjPCFillLevel",0);
    adjGMRESMaxIters    = readOptionOrDefault<label>(adjointOptions,"adjGMRESMaxIters",1000);
    adjGMRESAbsTol      = readOptionOrDefault<scalar>(adjointOptions,"adjGMRESAbsTol",1.0e-16);
    adjGMRESRelTol      = readOptionOrDefault<scalar>(adjointOptions,"adjGMRESRelTol",1.0e-6);
    readListOption< List<word> >
    (
        normalizeResiduals,
        adjointOptions,
        "normalizeResiduals"
    );
    readListOption< List<word> >
    (
        normalizeStates,
        adjointOptions,
        "normalizeStates"
    );
    maxResConLv4JacPCMat= readOptionOrDefault< HashTable<label> >(adjointOptions,"maxResConLv4JacPCMat",{});
    stateScaling        = readOptionOrDefault< HashTable<scalar> >(adjointOptions,"stateScaling",{});
    residualScaling     = readOptionOrDefault< HashTable<scalar> >(adjointOptions,"residualScaling",{});
    readListOption< List<word> >
    (
        adjDVTypes,
        adjointOptions,
        "adjDVTypes"
    );
    nFFDPoints          = readOptionOrDefault<label>(adjointOptions,"nFFDPoints",0);
    
    // read flow options from system/adjointDict or use default values
    dictionary flowOptions
    (
        adjointDict_.subDict
        (
            "flowOptions"
        )
    ); 
    setFlowBCs         = readBooleanOptionOrDefault(flowOptions,"setFlowBCs","false");
    flowBCs            = flowOptions.subDict("flowBCs");
    adjointParameters_.add("flowBCs",flowBCs);
    readListOption< List<word> >
    (
        inletPatches,
        flowOptions,
        "inletPatches"
    );
    readListOption< List<word> >
    (
        outletPatches,
        flowOptions,
        "outletPatches"
    );

    derivUInInfo         = flowOptions.subDict("derivUInInfo");
    adjointParameters_.add("derivUInInfo",derivUInInfo);
    userDefinedPatchInfo = flowOptions.subDict("userDefinedPatchInfo");
    adjointParameters_.add("userDefinedPatchInfo",userDefinedPatchInfo);
    userDefinedVolumeInfo= flowOptions.subDict("userDefinedVolumeInfo");
    adjointParameters_.add("userDefinedVolumeInfo",userDefinedVolumeInfo);
    referenceValues      = readOptionOrDefault< HashTable<scalar> >(flowOptions,"referenceValues",{});
    divDev2              = readBooleanOptionOrDefault(flowOptions,"divDev2","true");
    flowCondition        = readOptionOrDefault< word >(flowOptions,"flowCondition","Incompressible");
    // read transport and thermophysical properties
    if (flowCondition=="Incompressible")
    {
        IOdictionary transportDict
        (
            IOobject
            (
                "transportProperties",
                mesh_.time().constant(),
                mesh_,
                IOobject::MUST_READ,
                IOobject::NO_WRITE,
                false
            )
        );
        forAll(transportDict.toc(),idxI)
        {
            word key=transportDict.toc()[idxI];
            if (key!="transportModel")
            {
                scalar tmpVal = readScalar( transportDict.lookup(key) ); 
                flowProperties.set(key,tmpVal);
            }
        }
    }
    else if (flowCondition=="Compressible")
    {
        IOdictionary thermophysicalDict
        (
            IOobject
            (
                "thermophysicalProperties",
                mesh_.time().constant(),
                mesh_,
                IOobject::MUST_READ,
                IOobject::NO_WRITE,
                false
            )
        );
        forAll(thermophysicalDict.toc(),idxI)
        {
            word key=thermophysicalDict.toc()[idxI];
            if (key=="mixture")
            {
                dictionary subDict0=thermophysicalDict.subDict(key);
                forAll(subDict0.toc(),idxJ)
                {
                    word key1=subDict0.toc()[idxJ];
                    dictionary subDict1=subDict0.subDict(key1);
                    forAll(subDict1.toc(),idxK)
                    {
                        word key2=subDict1.toc()[idxK];
                        scalar tmpVal = readScalar( subDict1.lookup(key2) ); 
                        flowProperties.set(key2,tmpVal);
                    }
                }
            }
        }
        // read Prt from turbulenceProperties
        IOdictionary turbDict
        (
            IOobject
            (
                "turbulenceProperties",
                mesh_.time().constant(),
                mesh_,
                IOobject::MUST_READ,
                IOobject::NO_WRITE,
                false
            )
        );
        dictionary RASSubDict=turbDict.subDict("RAS");
        scalar Prt1= RASSubDict.lookupOrDefault<scalar>("Prt",1.0); 
        flowProperties.set("Prt",Prt1);

    }
    adjointParameters_.add("flowProperties",flowProperties);
    // NK options
    useNKSolver         = readBooleanOptionOrDefault(flowOptions,"useNKSolver","false");
    nkPseudoTransient   = readBooleanOptionOrDefault(flowOptions,"nkPseudoTransient","false");
    nkSegregatedTurb    = readBooleanOptionOrDefault(flowOptions,"nkSegregatedTurb","false");
    nkSegregatedPhi     = readBooleanOptionOrDefault(flowOptions,"nkSegregatedPhi","false");
    nkEWRTol0           = readOptionOrDefault< scalar >(flowOptions,"nkEWRTol0",0.3);
    nkEWRTolMax         = readOptionOrDefault< scalar >(flowOptions,"nkEWRTolMax",0.7);
    nkPCLag             = readOptionOrDefault< label >(flowOptions,"nkPCLag",5);
    nkGMRESRestart      = readOptionOrDefault< label >(flowOptions,"nkGMRESRestart",500);
    nkASMOverlap        = readOptionOrDefault< label >(flowOptions,"nkASMOverlap",1);
    nkPCFillLevel       = readOptionOrDefault< label >(flowOptions,"nkPCFillLevel",0);
    nkGlobalPCIters     = readOptionOrDefault<label>(flowOptions,"nkGlobalPCIters",0);
    nkLocalPCIters      = readOptionOrDefault<label>(flowOptions,"nkLocalPCIters",1);
    nkGMRESMaxIters     = readOptionOrDefault< label >(flowOptions,"nkGMRESMaxIters",500);
    nkJacMatReOrdering  = readOptionOrDefault< word >(flowOptions,"nkJacMatReOrdering","rcm");
    nkRelTol            = readOptionOrDefault< scalar >(flowOptions,"nkRelTol",1e-8);
    nkAbsTol            = readOptionOrDefault< scalar >(flowOptions,"nkAbsTol",1e-12);
    nkSTol              = readOptionOrDefault< scalar >(flowOptions,"nkSTol",1e-8);
    nkMaxIters          = readOptionOrDefault< label >(flowOptions,"nkMaxIters",100);
    nkMaxFuncEvals      = readOptionOrDefault< label >(flowOptions,"nkMaxFuncEvals",10000);


    // read actuator disk options from system/adjointDict or use default values
    dictionary actuatorDiskOptions
    (
        adjointDict_.subDict
        (
            "actuatorDiskOptions"
        )
    ); 
    actuatorActive       = readOptionOrDefault<label>(actuatorDiskOptions,"actuatorActive",0);
    actuatorAdjustThrust = readOptionOrDefault<label>(actuatorDiskOptions,"actuatorAdjustThrust",0);
    actuatorVolumeNames  = readOptionOrDefault< List<word> >(actuatorDiskOptions,"actuatorVolumeNames",{});
    actuatorThrustCoeff  = readOptionOrDefault<scalarList>(actuatorDiskOptions,"actuatorThrustCoeff",{});
    actuatorPOverD       = readOptionOrDefault<scalarList>(actuatorDiskOptions,"actuatorPOverD",{});
    actuatorRotationDir  = readOptionOrDefault< List<word> >(actuatorDiskOptions,"actuatorRotationDir",{});

    // read objective function options from system/adjointDict or use default values
    dictionary objOptions
    (    
        adjointDict_.subDict
        (
            "objectiveFunctionOptions"
        )
    );
    readListOption< List<word> >
    (
        objFuncs,
        objOptions,
        "objFuncs"
    );
    readListOption< List< List<word> > >
    (
        objFuncGeoInfo,
        objOptions,
        "objFuncGeoInfo"
    );
    if (objFuncs.size() != objFuncGeoInfo.size())
    {
        FatalErrorIn("")<<" size(objFuncs)!=size(objFuncGeoInfo) "
        <<abort(FatalError);
    }
    dragDir              = readOptionOrDefault<vector>(objOptions,"dragDir",vector::zero);
    liftDir              = readOptionOrDefault<vector>(objOptions,"liftDir",vector::zero);
    CofR                 = readOptionOrDefault<vector>(objOptions,"CofR",vector::zero);
    rotRad               = readOptionOrDefault<vector>(objOptions,"rotRad",vector::zero);

    // print all the parameters to screen    
    Info<<"Adjoint Parameters"<<adjointParameters_<<endl;

    // read fvSchemes
    IOdictionary fvSchemesDict
    (
        IOobject
        (
            "fvSchemes",
            mesh_.time().system(),
            mesh_,
            IOobject::MUST_READ,
            IOobject::NO_WRITE,
            false
        )
    );
    Info<<"fvSchemes"<<fvSchemesDict<<endl;

    // read fvSolution
    IOdictionary fvSolutionDict
    (
        IOobject
        (
            "fvSolution",
            mesh_.time().system(),
            mesh_,
            IOobject::MUST_READ,
            IOobject::NO_WRITE,
            false
        )
    );
    Info<<"fvSolution"<<fvSolutionDict<<endl;

}

AdjointIO::~AdjointIO()
{
}


// read a boolean option from dict and add them to adjointParameters_
label AdjointIO::readBooleanOptionOrDefault
(
    const dictionary& dict,
    word option,
    word defaultValue
)
{
    
    word value = dict.lookupOrDefault<word>(option,defaultValue); 
    if (value == defaultValue)
    {
        // notify the user the default value is used.
        word optionPrint = option+"(D)"; 
        adjointParameters_.add(optionPrint,value);
    }
    else
    {
        adjointParameters_.add(option,value);
    }
    
    label result(0);
    if(value=="true")
    {
        result=1;
    }
    return result;
}

void AdjointIO::readVectorBinary(const Vec vecIn, const word prefix) const
{

    std::ostringstream fileNameStream("");
    fileNameStream<<prefix<<".bin";
    word fileName = fileNameStream.str();
    
    PetscViewer    viewer;
    PetscViewerBinaryOpen(PETSC_COMM_WORLD,fileName.c_str(),FILE_MODE_READ,&viewer);
    VecLoad(vecIn,viewer);
    PetscViewerDestroy(&viewer);
    
    return;

}

void AdjointIO::writeVectorBinary(const Vec vecIn, const word prefix) const
{

    std::ostringstream fileNameStream("");
    fileNameStream<<prefix<<".bin";
    word fileName = fileNameStream.str();
    
    PetscViewer    viewer;
    PetscViewerBinaryOpen(PETSC_COMM_WORLD,fileName.c_str(),FILE_MODE_WRITE,&viewer);
    VecView(vecIn,viewer);
    PetscViewerDestroy(&viewer);
    
    return;

}

void AdjointIO::readMatrixBinary(const Mat matIn, const word prefix) const
{

    std::ostringstream fileNameStream("");
    fileNameStream<<prefix<<".bin";
    word fileName = fileNameStream.str();
    
    PetscViewer    viewer;
    PetscViewerBinaryOpen(PETSC_COMM_WORLD,fileName.c_str(),FILE_MODE_READ,&viewer);
    MatLoad(matIn,viewer);
    PetscViewerDestroy(&viewer);
    
    return;

}

void AdjointIO::writeMatrixBinary(const Mat matIn, const word prefix) const
{

    std::ostringstream fileNameStream("");
    fileNameStream<<prefix<<".bin";
    word fileName = fileNameStream.str();
    
    PetscViewer    viewer;
    PetscViewerBinaryOpen(PETSC_COMM_WORLD,fileName.c_str(),FILE_MODE_WRITE,&viewer);
    MatView(matIn,viewer);
    PetscViewerDestroy(&viewer);
    
    return;

}

void AdjointIO::writeMatrixASCII(const Mat matIn, const word prefix) const
{

    std::ostringstream fileNameStream("");
    fileNameStream<<prefix<<".dat";
    word fileName = fileNameStream.str();
    
    PetscViewer    viewer;
    PetscViewerASCIIOpen(PETSC_COMM_WORLD,fileName.c_str(),&viewer);
    MatView(matIn,viewer);
    PetscViewerDestroy(&viewer);
    
    return;

}

void AdjointIO::writeVectorASCII(const Vec vecIn, const word prefix) const
{
    std::ostringstream fileNameStream("");
    fileNameStream<<prefix<<".dat";
    word fileName = fileNameStream.str();
    
    PetscViewer    viewer;
    PetscViewerASCIIOpen(PETSC_COMM_WORLD,fileName.c_str(),&viewer);
    PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_MATLAB); // write all the digits
    VecView(vecIn,viewer);
    PetscViewerDestroy(&viewer);
    
    return;
}


void AdjointIO::writeMatRowSize(Mat matIn, word prefix) const
{
    //output the matrix to a file
    PetscInt Istart,Iend,colMin,colMaxp1;
    PetscInt nCols;
    const PetscInt    *cols;
    const PetscScalar *vals;
    Vec colCountOn,colCountOff;

    MatGetOwnershipRange(matIn,&Istart,&Iend);
    MatGetOwnershipRangeColumn(matIn,&colMin,&colMaxp1);
    
    label nLocalRows = Iend - Istart;
    
    VecCreate(PETSC_COMM_WORLD,&colCountOn);
    VecSetSizes(colCountOn,nLocalRows,PETSC_DECIDE);
    VecSetFromOptions(colCountOn);
    
    VecDuplicate(colCountOn,&colCountOff);
    
    for(label i=Istart; i<Iend; i++)
    {
        MatGetRow(matIn, i,&nCols,&cols,&vals);
        PetscInt onCounter = 0;
        PetscInt offCounter =0;
        for (label j=0; j<nCols; j++)
        {
            if(fabs(vals[j])>1.0e-10)
            {
                // count this row
                if(cols[j]>=colMin and cols[j]<colMaxp1)
                {
                    onCounter++; 
                }
                else
                {
                    offCounter++; 
                }
            }
        }
        MatRestoreRow(matIn, i,&nCols,&cols,&vals);
        
        VecSetValue(colCountOn,i,onCounter,INSERT_VALUES);
        VecSetValue(colCountOff,i,offCounter,INSERT_VALUES);
    }
    VecAssemblyBegin(colCountOn);
    VecAssemblyEnd(colCountOn);
    VecAssemblyBegin(colCountOff);
    VecAssemblyEnd(colCountOff);

    PetscViewer    viewer;
    string onVecSizeFile=prefix+"_OnSize.txt";
    PetscViewerASCIIOpen(PETSC_COMM_WORLD,onVecSizeFile.c_str(),&viewer);
    VecView(colCountOn,viewer);
    string offVecSizeFile=prefix+"_OffSize.txt";
    PetscViewerASCIIOpen(PETSC_COMM_WORLD,offVecSizeFile.c_str(),&viewer);
    VecView(colCountOff,viewer);
    
    PetscViewerDestroy(&viewer);

    VecDestroy(&colCountOn);
    VecDestroy(&colCountOff);

}

word AdjointIO::getPName() const
{
    word pName="p";
    IOdictionary controlDict
    (
        IOobject
        (
            "controlDict",
            mesh_.time().system(),
            mesh_,
            IOobject::MUST_READ,
            IOobject::NO_WRITE,
            false
        )
    );
    word solverName = word(controlDict.lookup("application"));

    // create word regular expression for buoyant solvers
    wordReList buoyantSolverWordReList
    {
        {"buoyant.*", wordRe::REGEX}
    };
    wordRes buoyantSolverWordRes(buoyantSolverWordReList);

    if (buoyantSolverWordRes(solverName))
    {
        pName="p_rgh";
    }
    return pName;
}

void AdjointIO::boundVar(volScalarField& var)
{
    const scalar vGreat = 1e200;
    label useUpperBound = 0, useLowerBound=0;

    const dictionary& simple = mesh_.solutionDict().subDict("SIMPLE");

    scalar varMin = simple.lookupOrDefault<scalar>(var.name()+"LowerBound",-vGreat); 
    scalar varMax = simple.lookupOrDefault<scalar>(var.name()+"UpperBound",vGreat); 

    
    forAll(var,cellI)
    {
        if(var[cellI]<=varMin) 
        {
            var[cellI]=varMin;
            useLowerBound = 1;
        }
        if(var[cellI]>=varMax) 
        {
            var[cellI]=varMax;
            useUpperBound = 1;
        }
    }

    forAll(var.boundaryField(),patchI)
    {
        forAll(var.boundaryField()[patchI],faceI)
        {
            if(var.boundaryFieldRef()[patchI][faceI]<=varMin) 
            {
                var.boundaryFieldRef()[patchI][faceI]=varMin;
                useLowerBound = 1;
            }
            if(var.boundaryFieldRef()[patchI][faceI]>=varMax) 
            {
                var.boundaryFieldRef()[patchI][faceI]=varMax;
                useUpperBound = 1;
            }
        }
    }

    if (useUpperBound) Info<<"Bounding "<<var.name()<<"<"<<varMax<<endl;
    if (useLowerBound) Info<<"Bounding "<<var.name()<<">"<<varMin<<endl;

    return;
}

void AdjointIO::boundVar(volVectorField& var)
{
    const scalar vGreat = 1e200;
    label useUpperBound = 0, useLowerBound=0;

    const dictionary& simple = mesh_.solutionDict().subDict("SIMPLE");

    scalar varMin = simple.lookupOrDefault<scalar>(var.name()+"LowerBound",-vGreat); 
    scalar varMax = simple.lookupOrDefault<scalar>(var.name()+"UpperBound",vGreat); 

    
    forAll(var,cellI)
    {
        for(label i=0;i<3;i++)
        {
            if(var[cellI][i]<=varMin) 
            {
                var[cellI][i]=varMin;
                useLowerBound = 1;
            }
            if(var[cellI][i]>=varMax) 
            {
                var[cellI][i]=varMax;
                useUpperBound = 1;
            }
        }
    }

    forAll(var.boundaryField(),patchI)
    {
        forAll(var.boundaryField()[patchI],faceI)
        {
            for(label i=0;i<3;i++)
            {
                if(var.boundaryFieldRef()[patchI][faceI][i]<=varMin) 
                {
                    var.boundaryFieldRef()[patchI][faceI][i]=varMin;
                    useLowerBound = 1;
                }
                if(var.boundaryFieldRef()[patchI][faceI][i]>=varMax) 
                {
                    var.boundaryFieldRef()[patchI][faceI][i]=varMax;
                    useUpperBound = 1;
                }
            }
        }
    }

    if (useUpperBound) Info<<"Bounding "<<var.name()<<"<"<<varMax<<endl;
    if (useLowerBound) Info<<"Bounding "<<var.name()<<">"<<varMin<<endl;

    return;
}

void AdjointIO::calcUabs(const volVectorField& Urel,volVectorField& Uabs) const
{
    dimensionedVector rotRad("rotRad", dimless/dimTime,this->rotRad);
    dimensionedVector rotCofR("rotCofR", dimLength,this->CofR);

    Uabs=Urel+( rotRad^(mesh_.C()-rotCofR) );

    forAll(Uabs.boundaryField(),patchI)
    {
        forAll(Uabs.boundaryField()[patchI],faceI)
        {
            Uabs.boundaryFieldRef()[patchI][faceI] = 
                Urel.boundaryField()[patchI][faceI]+
                ( 
                    rotRad.value()^( mesh_.C().boundaryField()[patchI][faceI]-rotCofR.value() )  
                );
        }
    }
}

void AdjointIO::setFlowBoundaryConditions()
{
    /*
    A general function to read the inlet/outlet values from adjointDict, set
    the corresponding values to the boundary field, and write
    the update field to files.
    it also setup turbulence wall boundary condition
    Note: this function should be called right after all the
    volFields are loaded
    If nothing is set, the BC will remain unchanged
    Example
    flowBCs 
    { 
        bc1
        {
            patch inlet; 
            variable U; 
            value {10 0 0};
        } 
        bc2
        {
            patch outlet;
            variable p;
            value {0};
        }
        useWallFunction 1;
    }
    */
    
    const objectRegistry& db = mesh_.thisDb();

    label setTurbWallBCs=0;
    label useWallFunction=0;

    forAll(flowBCs.toc(),idxI)
    {
        word BCs=flowBCs.toc()[idxI];

        if (BCs=="useWallFunction")
        {
            setTurbWallBCs=1;
            word tmp=word(flowBCs.lookup("useWallFunction"));
            if (tmp=="true") useWallFunction=1;
            else if (tmp=="false") useWallFunction=0;
            else FatalErrorIn("")<<"useWallFunction can be either true or false"<<abort(FatalError);
            continue;
        }

        dictionary BCsSub = flowBCs.subDict(BCs);

        word patch = word(BCsSub["patch"]);
        word variable = word(BCsSub["variable"]);
        scalarList value = BCsSub["value"];

        if (value.size()==1)
        {
            if (!db.foundObject<volScalarField>(variable))
            {
                //Info<<variable<<" not found, skip it."
                continue;
            }
            // it is a scalar
            volScalarField& state                                            
            (                                                                      
                const_cast<volScalarField&>                                             
                (                                                                  
                    db.lookupObject<volScalarField>(variable)                         
                )                                                                  
            ); 

            Info<<"Setting BC based on adjointDict..."<<endl;
            Info<<"Setting "<<variable<<" = "<<value[0]<<" at "<<patch<<endl;
        
            label patchI = mesh_.boundaryMesh().findPatchID( patch );
            
            // for decomposed domain, don't set BC if the patch is empty 
            if (mesh_.boundaryMesh()[patchI].size()>0)
            {
                if(state.boundaryFieldRef()[patchI].type()=="fixedValue")
                {
                    forAll(state.boundaryFieldRef()[patchI],faceI)
                    {
                        state.boundaryFieldRef()[patchI][faceI] = value[0];  
                    }
                }
                else if (state.boundaryFieldRef()[patchI].type()=="inletOutlet" ||
                         state.boundaryFieldRef()[patchI].type()=="outletInlet")
                {
                    // set value
                    forAll(state.boundaryFieldRef()[patchI],faceI)
                    {
                        state.boundaryFieldRef()[patchI][faceI] = value[0]; 
                    }
                    // set inletValue
                    mixedFvPatchField<scalar>& inletOutletPatch = 
                        refCast< mixedFvPatchField<scalar> > (state.boundaryFieldRef()[patchI]);
                    inletOutletPatch.refValue()=value[0]; 
                }
                else if (state.boundaryFieldRef()[patchI].type()=="fixedGradient")
                {
                    fixedGradientFvPatchScalarField& patchBC =
                        refCast< fixedGradientFvPatchScalarField > (state.boundaryFieldRef()[patchI]);
                    scalarField& grad = const_cast<scalarField&>(patchBC.gradient());
                    forAll(grad,idxI)
                    {
                        grad[idxI] = value[0];
                    }
                }
                else
                {
                    FatalErrorIn("")<<"only support fixedValues, inletOutlet, outletInlet, fixedGradient!"<<abort(FatalError);
                }
            }
            
            state.write();

        }
        else if (value.size()==3)
        {
            if (!db.foundObject<volVectorField>(variable))
            {
                //Info<<variable<<" not found, skip it."
                continue;
            }
            // it is a vector
            volVectorField& state                                            
            (                                                                      
                const_cast<volVectorField&>                                             
                (                                                                  
                    db.lookupObject<volVectorField>(variable)                         
                )                                                                  
            ); 

            vector valVec={value[0],value[1],value[2]};

            Info<<"Setting BC based on adjointDict..."<<endl;
            Info<<"Setting "<<variable<<" = ("<<value[0]<<" "<<value[1]<<" "<<value[2]<<") at "<<patch<<endl;
        
            label patchI = mesh_.boundaryMesh().findPatchID( patch );
            
            // for decomposed domain, don't set BC if the patch is empty 
            if (mesh_.boundaryMesh()[patchI].size()>0)
            {
                if(state.boundaryFieldRef()[patchI].type()=="fixedValue")
                {
                    forAll(state.boundaryFieldRef()[patchI],faceI)
                    {
                        state.boundaryFieldRef()[patchI][faceI] = valVec;  
                    }
                }
                else if (state.boundaryFieldRef()[patchI].type()=="inletOutlet" ||
                         state.boundaryFieldRef()[patchI].type()=="outletInlet" )
                {
                    // set value
                    forAll(state.boundaryFieldRef()[patchI],faceI)
                    {
                        state.boundaryFieldRef()[patchI][faceI] = valVec;  
                    }
                    // set inletValue
                    mixedFvPatchField<vector>& inletOutletPatch = 
                        refCast< mixedFvPatchField<vector> > (state.boundaryFieldRef()[patchI]);
                    inletOutletPatch.refValue()=valVec;
                }
                else if (state.boundaryFieldRef()[patchI].type()=="tractionDisplacement")
                {
                    tractionDisplacementFvPatchVectorField& patchBC = 
                        refCast< tractionDisplacementFvPatchVectorField > (state.boundaryFieldRef()[patchI]);
                    vectorField& traction = const_cast<vectorField&>(patchBC.traction());
                    scalarField& pressure = const_cast<scalarField&>(patchBC.pressure());
                    scalarList pValue = BCsSub["pressure"];
                    if (pValue.size()!=1) FatalErrorIn("")<<"pressure size should be 1"<<abort(FatalError);

                    forAll(traction,idxI)
                    {
                        traction[idxI][0] = value[0];
                        traction[idxI][1] = value[1];
                        traction[idxI][2] = value[2];
                    }
                    forAll(pressure,idxI)
                    {
                        pressure[idxI] = pValue[0];
                    }
                }
                else if (state.boundaryFieldRef()[patchI].type()=="fixedGradient")
                {
                    fixedGradientFvPatchVectorField& patchBC =
                        refCast< fixedGradientFvPatchVectorField > (state.boundaryFieldRef()[patchI]);
                    vectorField& grad = const_cast<vectorField&>(patchBC.gradient());
                    forAll(grad,idxI)
                    {
                        grad[idxI][0] = value[0];
                        grad[idxI][1] = value[1];
                        grad[idxI][2] = value[2];
                    }
                }
                else
                {
                    FatalErrorIn("")<<"only support fixedValues, inletOutlet, fixedGradient, and tractionDisplacement!"<<abort(FatalError);
                }
            }
            
            state.write();
        }
        else
        {
            FatalErrorIn("")<<"value should be a list of either 1 (scalar) or 3 (vector) elements"<<abort(FatalError);
        }

    }
 

    // we also set wall boundary conditions for turbulence variables
    if(setTurbWallBCs)
    {
        wordList turbVars={"nut","nuTilda","k","omega","epsilon"};

        IOdictionary turbDict
        (
            IOobject
            (
                "turbulenceProperties",
                mesh_.time().constant(),
                mesh_,
                IOobject::MUST_READ,
                IOobject::NO_WRITE,
                false
            )
        );

        dictionary coeffDict(turbDict.subDict("RAS"));
        word turbModelType = word(coeffDict["RASModel"]);
        //Info<<"turbModelType: "<<turbModelType<<endl;
    
        // create word regular expression for SA model
        wordReList SAModelWordReList
        {
            {"SpalartAllmaras.*", wordRe::REGEX}
        };
        wordRes SAModelWordRes(SAModelWordReList);
    
        forAll(turbVars,idxI)
        {
            word turbVar=turbVars[idxI];

            // ------ nut ----------
            if(turbVar == "nut" && db.foundObject<volScalarField>("nut"))
            {
                volScalarField& nut                                          
                (                                                                      
                    const_cast<volScalarField&>                                             
                    (                                                                  
                        db.lookupObject<volScalarField>("nut")                         
                    )                                                                  
                );

                forAll(nut.boundaryField(),patchI)
                {
                    if (mesh_.boundaryMesh()[patchI].type()=="wall")
                    {
                        Info<<"Setting nut wall BC for "<<mesh_.boundaryMesh()[patchI].name()<<". ";

                        if ( useWallFunction  )
                        {
                            // wall function for SA
                            if (SAModelWordRes(turbModelType))
                            {
                                nut.boundaryFieldRef().set
                                (
                                    patchI,
                                    fvPatchField<scalar>::New("nutUSpaldingWallFunction", mesh_.boundary()[patchI], nut)
                                );
                                Info<<"BCType=nutUSpaldingWallFunction"<<endl;
                            }
                            else // wall function for kOmega and kEpsilon
                            {
                                nut.boundaryFieldRef().set
                                (
                                    patchI,
                                    fvPatchField<scalar>::New("nutkWallFunction", mesh_.boundary()[patchI], nut)
                                );
                                Info<<"BCType=nutkWallFunction"<<endl;
                            }

                            // set boundary values
                            // for decomposed domain, don't set BC if the patch is empty 
                            if (mesh_.boundaryMesh()[patchI].size()>0)
                            {
                                scalar wallVal=nut[0];
                                forAll(nut.boundaryFieldRef()[patchI],faceI)
                                {
                                    nut.boundaryFieldRef()[patchI][faceI] = wallVal; // assign uniform field
                                }
                            }
                        }
                        else 
                        {
                            nut.boundaryFieldRef().set
                            (
                                patchI,
                                fvPatchField<scalar>::New("nutLowReWallFunction", mesh_.boundary()[patchI], nut)
                            );
                            Info<<"BCType=nutLowReWallFunction"<<endl;

                            // set boundary values
                            // for decomposed domain, don't set BC if the patch is empty 
                            if (mesh_.boundaryMesh()[patchI].size()>0)
                            {
                                forAll(nut.boundaryFieldRef()[patchI],faceI)
                                {
                                    nut.boundaryFieldRef()[patchI][faceI] = 1e-14; // assign uniform field
                                }
                            }
                        }
                    }
                }

                nut.write();
            }
    
            // ------ k ----------
            if(turbVar == "k" && db.foundObject<volScalarField>("k") )
            {
                volScalarField& k                                          
                (                                                                      
                    const_cast<volScalarField&>                                             
                    (                                                                  
                        db.lookupObject<volScalarField>("k")                         
                    )                                                                  
                );
                forAll(k.boundaryField(),patchI)
                {
                    if (mesh_.boundaryMesh()[patchI].type()=="wall")
                    {
                        Info<<"Setting k wall BC for "<<mesh_.boundaryMesh()[patchI].name()<<". ";

                        if ( useWallFunction  )
                        {
                            k.boundaryFieldRef().set
                            (
                                patchI,
                                fvPatchField<scalar>::New("kqRWallFunction", mesh_.boundary()[patchI], k)
                            );
                            Info<<"BCType=kqRWallFunction"<<endl;

                            // set boundary values
                            // for decomposed domain, don't set BC if the patch is empty 
                            if (mesh_.boundaryMesh()[patchI].size()>0)
                            {
                                scalar wallVal=k[0];
                                forAll(k.boundaryFieldRef()[patchI],faceI)
                                {
                                    k.boundaryFieldRef()[patchI][faceI] = wallVal; // assign uniform field
                                }
                            }
                        }
                        else 
                        {
                            k.boundaryFieldRef().set
                            (
                                patchI,
                                fvPatchField<scalar>::New("fixedValue", mesh_.boundary()[patchI], k)
                            );
                            Info<<"BCType=fixedValue"<<endl;

                            // set boundary values
                            // for decomposed domain, don't set BC if the patch is empty 
                            if (mesh_.boundaryMesh()[patchI].size()>0)
                            {
                                forAll(k.boundaryFieldRef()[patchI],faceI)
                                {
                                    k.boundaryFieldRef()[patchI][faceI] = 1e-14; // assign uniform field
                                }
                            }
                        }
                    }
                }

                k.write();

            }
    
            // ------ omega ----------
            if(turbVar == "omega" && db.foundObject<volScalarField>("omega") )
            {
                volScalarField& omega                                          
                (                                                                      
                    const_cast<volScalarField&>                                             
                    (                                                                  
                        db.lookupObject<volScalarField>("omega")                         
                    )                                                                  
                );
                forAll(omega.boundaryField(),patchI)
                {
                    if (mesh_.boundaryMesh()[patchI].type()=="wall")
                    {
                        Info<<"Setting omega wall BC for "<<mesh_.boundaryMesh()[patchI].name()<<". ";

                        omega.boundaryFieldRef().set
                        (
                            patchI,
                            fvPatchField<scalar>::New("omegaWallFunction", mesh_.boundary()[patchI], omega)
                        );
                        Info<<"BCType=omegaWallFunction"<<endl;

                        // set boundary values
                        // for decomposed domain, don't set BC if the patch is empty 
                        if (mesh_.boundaryMesh()[patchI].size()>0)
                        {
                            scalar wallVal=omega[0];
                            forAll(omega.boundaryFieldRef()[patchI],faceI)
                            {
                                omega.boundaryFieldRef()[patchI][faceI] = wallVal; // assign uniform field
                            }
                        }

                    }
                }
                
                omega.write();
            }
    
            // ------ epsilon ----------
            if(turbVar == "epsilon" && db.foundObject<volScalarField>("epsilon") )
            {
                volScalarField& epsilon                                          
                (                                                                      
                    const_cast<volScalarField&>                                             
                    (                                                                  
                        db.lookupObject<volScalarField>("epsilon")                         
                    )                                                                  
                );
                forAll(epsilon.boundaryField(),patchI)
                {
                    if (mesh_.boundaryMesh()[patchI].type()=="wall")
                    {
                        Info<<"Setting epsilon wall BC for "<<mesh_.boundaryMesh()[patchI].name()<<". ";

                        if ( useWallFunction  )
                        {
                            epsilon.boundaryFieldRef().set
                            (
                                patchI,
                                fvPatchField<scalar>::New("epsilonWallFunction", mesh_.boundary()[patchI], epsilon)
                            );
                            Info<<"BCType=epsilonWallFunction"<<endl;

                            // set boundary values
                            // for decomposed domain, don't set BC if the patch is empty 
                            if (mesh_.boundaryMesh()[patchI].size()>0)
                            {
                                scalar wallVal=epsilon[0];
                                forAll(epsilon.boundaryFieldRef()[patchI],faceI)
                                {
                                    epsilon.boundaryFieldRef()[patchI][faceI] = wallVal; // assign uniform field
                                }
                            }
                        }
                        else 
                        {
                            epsilon.boundaryFieldRef().set
                            (
                                patchI,
                                fvPatchField<scalar>::New("fixedValue", mesh_.boundary()[patchI], epsilon)
                            );
                            Info<<"BCType=fixedValue"<<endl;

                            // set boundary values
                            // for decomposed domain, don't set BC if the patch is empty 
                            if (mesh_.boundaryMesh()[patchI].size()>0)
                            {
                                forAll(epsilon.boundaryFieldRef()[patchI],faceI)
                                {
                                    epsilon.boundaryFieldRef()[patchI][faceI] = 1e-14; // assign uniform field
                                }
                            }
                        }
                    }
                }
                
                epsilon.write();
            }
    
        }
    }
  
    return;

}




// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
