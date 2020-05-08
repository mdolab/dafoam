/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v1.0

\*---------------------------------------------------------------------------*/

#include "AdjointObjectiveFunction.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

// Constructors
AdjointObjectiveFunction::AdjointObjectiveFunction
(
    const fvMesh& mesh,
    const AdjointIO& adjIO,
    AdjointRASModel& adjRAS,
    AdjointIndexing& adjIdx,
    AdjointJacobianConnectivity& adjCon
)
    :
    mesh_(mesh),
    adjIO_(adjIO),
    adjRAS_(adjRAS),
    adjIdx_(adjIdx),
    adjCon_(adjCon),
    NUSField_
    (
        IOobject
        (
            "NUS",
            mesh.time().timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        mesh,
        dimensionedScalar("NUS",dimensionSet(0,0,0,0,0,0,0),0.0),
        fixedValueFvPatchScalarField::typeName
    ),
    TBulk_
    (
        IOobject
        (
            "TBulk",
            mesh.time().timeName(),
            mesh,
            IOobject::READ_IF_PRESENT,
            IOobject::NO_WRITE
        ),
        mesh,
        dimensionedScalar("TBulk",dimensionSet(0,0,0,1,0,0,0),adjIO_.flowProperties["TRef"]),
        fixedValueFvPatchScalarField::typeName
    ),
    Pr_(adjIO_.flowProperties["Pr"])
{

    // Register all possible objective functions and give
    // them a ID according to their ordering
    objFuncsReg_ = {"CD","CL","CMX","CMY","CMZ","CPL","NUS","AVGV","VARV","AVGS","VMS","MFR","TPR","TTR"};
    forAll(objFuncsReg_,idxI) 
    {
        word objFunc = objFuncsReg_[idxI];
        objFuncsRegID_.set(objFunc,idxI);
    }

    // now initialize all the obj function lists
    label nObjFuncsReg = objFuncsReg_.size();
    objFuncsVal_.setSize(nObjFuncsReg);
    objFuncsValRef_.setSize(nObjFuncsReg);
    objFuncsPartialDeriv_.setSize(nObjFuncsReg);
    objFuncsDiscreteVal_.setSize(nObjFuncsReg);
    objFuncsDiscreteValRef_.setSize(nObjFuncsReg);
    objFuncsDiscretePartialDeriv_.setSize(nObjFuncsReg);

    // get the user specified objective functions and their patches for adjoint
    // Note: we may set different patches for each objective function
    objFuncs_ = adjIO_.objFuncs;
    objFuncGeoInfo_ = adjIO_.objFuncGeoInfo;
    
    // check if the user specified obj funcs are registered
    forAll(objFuncs_,idxI)
    {
        if(!adjIO_.isInList<word>(objFuncs_[idxI],objFuncsReg_))
        {
            FatalErrorIn(" ")<<"objFunc "<<objFuncs_[idxI]<<" not valid"<< abort(FatalError);   
        }
    }

    // this list contains all the force related obj funcs. If any objFunc in this list
    // is specified in adjointDict, we will initialize the surfForces_
    forceRelatedObjFuncs_={"CD","CL","CMX","CMY","CMZ"};
    
    // initialize surface force if any objFunc is in forceRelatedObjFuncs
    this->initializeSurfForces();
    
    // initialize forcePatches_, which will be used in calcForces
    this->initializeForcePatches();


    // now allocate the surf variables for all of the cost functions so
    // that we can use coloring to evaluate the objective partials 
    forAll(objFuncs_,idxI)
    {
        word objFunc = objFuncs_[idxI];
        label nGeoElements = adjIdx_.getNLocalObjFuncGeoElements(objFunc);
        label objFuncID = objFuncsRegID_[objFunc];

        objFuncsDiscreteVal_[objFuncID].setSize(nGeoElements);
        objFuncsDiscreteValRef_[objFuncID].setSize(nGeoElements);
        objFuncsDiscretePartialDeriv_[objFuncID].setSize(nGeoElements);
    }

    IOdictionary thermalProperties
    (
        IOobject
        (
            "thermalProperties",
            mesh_.time().constant(),
            mesh_,
            IOobject::MUST_READ,
            IOobject::NO_WRITE
        )
    );
    Switch thermalStress1( thermalProperties.lookup("thermalStress"));
    thermalStress = thermalStress1;
}

AdjointObjectiveFunction::~AdjointObjectiveFunction()
{
}


vector AdjointObjectiveFunction::calcForces()
{

    // this code is pulled from: 
    // src/functionObjects/forcces/forces.C 
    // modified slightly
       
    const objectRegistry& db = mesh_.thisDb();
    const volScalarField& p = db.lookupObject<volScalarField>("p");

    // initialize the surfForces
    forAll(surfForces_,idxI)
    {
        surfForces_[idxI].x()=0;
        surfForces_[idxI].y()=0;
        surfForces_[idxI].z()=0;
    }
    
    const surfaceVectorField::Boundary& Sfb = mesh_.Sf().boundaryField();
    
    tmp<volSymmTensorField> tdevRhoReff = adjRAS_.devRhoReff();
    const volSymmTensorField::Boundary& devRhoReffb = tdevRhoReff().boundaryField();

    label cfsurfCounter = 0;
    forAll(forcePatches_,cI)
    {
        // get the patch id label
        label patchI = mesh_.boundaryMesh().findPatchID( forcePatches_[cI] );
        // create a shorter handle for the boundary patch
        const fvPatch& patch = mesh_.boundary()[patchI];
        // normal force
        vectorField fN
        (
            Sfb[patchI]*p.boundaryField()[patchI]
        );
        // tangential force
        vectorField fT(Sfb[patchI] & devRhoReffb[patchI]);
        // sum them up
        forAll(patch,faceI)
        {
            surfForces_[cfsurfCounter].x() = fN[faceI].x() + fT[faceI].x();
            surfForces_[cfsurfCounter].y() = fN[faceI].y() + fT[faceI].y();
            surfForces_[cfsurfCounter].z() = fN[faceI].z() + fT[faceI].z();
            cfsurfCounter+=1;
        }
    }

    vector fSum(vector::zero);
    forAll(surfForces_,idxI)
    {
        fSum += surfForces_[idxI];
    }

    for(label i=0; i<3; i++)
    {
        reduce(fSum[i], sumOp<scalar>() );
    }
    
    //Info<<"Force "<<fSum<<endl;

    return fSum;

}

void AdjointObjectiveFunction::calcNUS(scalar& objVal, scalarList& objDiscreteVal)
{ 
    // calculate Nusselt number
    // NUS = sum( alphaEff/alpha*gradT*LRef/deltaTRef*dS ) / sum(dS)
    
    // initialize the surfNUS
    wordList nusPatches;
    forAll(objFuncs_,idxI)
    {
        if(objFuncs_[idxI]=="NUS") nusPatches=objFuncGeoInfo_[idxI];
    }

    const objectRegistry& db = mesh_.thisDb();
    const volScalarField& T = db.lookupObject<volScalarField>("T");
    
    volScalarField alpha = adjRAS_.getAlpha();
    volScalarField alphaEff = adjRAS_.alphaEff();
    volScalarField alphaNorm = alphaEff/alpha;
    
    // calculate the area of all the obj func patches
    scalar Asum=0.0;
    forAll(nusPatches,cI)
    {
        // get the patch id label
        label patchI = mesh_.boundaryMesh().findPatchID( nusPatches[cI] );
        // create a shorter handle for the boundary patch
        const fvPatch& patch = mesh_.boundary()[patchI];
        forAll(patch,faceI)
        {
            Asum += mesh_.magSf().boundaryField()[patchI][faceI];
        }
    }
    reduce(Asum, sumOp<scalar>() );

    scalar LRef=adjIO_.referenceValues["LRef"];
        
    // calculate surfNUS
    label cfsurfCounter = 0;
    forAll(nusPatches,cI)
    {
        // get the patch id label
        label patchI = mesh_.boundaryMesh().findPatchID( nusPatches[cI] );
        // create a shorter handle for the boundary patch
        const fvPatch& patch = mesh_.boundary()[patchI];    

        const UList<label>& pFaceCells = mesh_.boundaryMesh()[patchI].faceCells();

        forAll(patch,faceI)
        {
            const label& idxN = pFaceCells[faceI];   
            scalar normDist =  adjRAS_.getNearWallDist(patchI,faceI);
            if(normDist==0) FatalErrorIn(" ")<<"normDist=0!"<< abort(FatalError); 
            scalar gradT = ( T.boundaryField()[patchI][faceI] - T[idxN] ) / normDist;

            scalar deltaT = T.boundaryField()[patchI][faceI] - TBulk_.boundaryField()[patchI][faceI];
            if(deltaT==0) FatalErrorIn(" ")<<"TRef-TInf=0!"<< abort(FatalError); 
            
            const scalar& alphaNormSurf = alphaNorm.boundaryField()[patchI][faceI];
            
            const scalar& magSfSurf = mesh_.magSf().boundaryField()[patchI][faceI];
                        
            scalar nusSurf  = alphaNormSurf*gradT*LRef/deltaT;
            scalar nusSurfAvg = nusSurf*magSfSurf/Asum;

            objDiscreteVal[cfsurfCounter] = nusSurfAvg;
            
            NUSField_.boundaryFieldRef()[patchI][faceI] = nusSurf;
            
            cfsurfCounter+=1;
        }
    }
    
    objVal=0.0;
    forAll(objDiscreteVal,idxI)
    {
        objVal += objDiscreteVal[idxI];
    }

    reduce(objVal, sumOp<scalar>() );
    
    return;
    
}

void AdjointObjectiveFunction::calcCPL(scalar& objVal, scalarList& objDiscreteVal)
{
    // CPL is the total pressure difference between inlet and outlet
    // normalized by 0.5*rhoRef*magURef*magURef
    // here we use area weighted averaging for total pressure
    // pTotal = sum{ ( p+0.5*rho*uMag*uMag ) * dS } / sum{dS}
    
    wordList cplPatches;
    forAll(objFuncs_,idxI)
    {
        if(objFuncs_[idxI]=="CPL") cplPatches=objFuncGeoInfo_[idxI];
    }
        
    const objectRegistry& db = mesh_.thisDb();
    const volScalarField& p = db.lookupObject<volScalarField>("p");
    const volVectorField& U = db.lookupObject<volVectorField>("U");
    volScalarField rho  = adjRAS_.getRho();
    
    // calc the inlet and outlet area
    scalar SfInlet=0.0;
    scalar SfOutlet=0.0;
    forAll(cplPatches,idxI)
    {
        word patchName = cplPatches[idxI];
        // for CPL we can set the inlet/outlet patch to be an UserDefined patch
        if( adjIdx_.isUserDefinedPatch(patchName) )
        {
            labelList userDefinedPatchFaces = adjIdx_.faceIdx4UserDefinedPatches[patchName];
            dictionary patchDict = adjIO_.userDefinedPatchInfo.subDict(patchName);
            scalar scale = readScalar( patchDict["scale"] );

            forAll(userDefinedPatchFaces,idxJ)
            {
                label faceI = userDefinedPatchFaces[idxJ];
                scalar magSf;
                // UserDefined face may be on a boundary patch for decomposed domains
                if(faceI < adjIdx_.nLocalInternalFaces) 
                {
                    magSf=mesh_.magSf()[faceI];
                }
                else
                {
                    label relIdx=faceI-adjIdx_.nLocalInternalFaces;
                    label patchIdx=adjIdx_.bFacePatchI[relIdx];
                    label faceIdx=adjIdx_.bFaceFaceI[relIdx];
                    magSf=mesh_.magSf().boundaryField()[patchIdx][faceIdx];
                }

                if( scale == 1.0 ) SfInlet += magSf;
                else if ( scale == -1.0 ) SfOutlet += magSf;
                else FatalErrorIn(" ")<<"userDefinedPatch scale not valid"<< abort(FatalError);   
            }
        }
        else
        {
            label patchI = mesh_.boundaryMesh().findPatchID( patchName );
            forAll(mesh_.boundaryMesh()[patchI],faceI)
            {
                if(adjIO_.isInList<word>(patchName,adjIO_.inletPatches) )
                {
                    SfInlet += mesh_.magSf().boundaryField()[patchI][faceI];
                }
                else if (adjIO_.isInList<word>(patchName,adjIO_.outletPatches) )
                {
                    SfOutlet += mesh_.magSf().boundaryField()[patchI][faceI];
                }
                else
                {
                    FatalErrorIn(" ")<<"CPL patch not valid"<< abort(FatalError);   
                }
            }
        }
        
    }
    reduce(SfInlet, sumOp<scalar>() );
    reduce(SfOutlet, sumOp<scalar>() );

    scalar magURef=adjIO_.referenceValues["magURef"];
    scalar rhoRef=adjIO_.referenceValues["rhoRef"];
    scalar pDyn = 0.5*rhoRef*magURef*magURef;

    // pTotal at inlet
    label cfsurfCounter = 0;
    forAll(cplPatches,idxI)
    {
        word patchName = cplPatches[idxI];
        // for CPL we can set the inlet/outlet patch to be an user defined patch
        if( adjIdx_.isUserDefinedPatch(patchName) )
        {
            surfaceScalarField pInterp = fvc::interpolate(p);
            surfaceVectorField UInterp = fvc::interpolate(U);
            surfaceScalarField rhoInterp = fvc::interpolate(rho);
                        
            labelList userDefinedPatchFaces = adjIdx_.faceIdx4UserDefinedPatches[patchName];

            dictionary patchDict = adjIO_.userDefinedPatchInfo.subDict(patchName);

            scalar scale = readScalar( patchDict["scale"] ); 
            
            forAll(userDefinedPatchFaces,idxJ)
            {
                label faceI = userDefinedPatchFaces[idxJ];
                
                scalar pS,magU,SfX,rhoS;
                
                if(faceI < adjIdx_.nLocalInternalFaces) 
                {
                    pS = pInterp[faceI];
                    magU = mag( UInterp[faceI] );
                    SfX = mesh_.magSf()[faceI];
                    rhoS = rhoInterp[faceI];
                }
                else
                {
                    label relIdx=faceI-adjIdx_.nLocalInternalFaces;
                    label patchIdx=adjIdx_.bFacePatchI[relIdx];
                    label faceIdx=adjIdx_.bFaceFaceI[relIdx];
                    pS = pInterp.boundaryField()[patchIdx][faceIdx];
                    magU = mag( UInterp.boundaryField()[patchIdx][faceIdx] );
                    SfX = mesh_.magSf().boundaryField()[patchIdx][faceIdx];
                    rhoS = rho.boundaryField()[patchIdx][faceIdx];
                }
                scalar pT = pS+0.5*rhoS*magU*magU;
                if( scale == 1.0)
                {
                    objDiscreteVal[cfsurfCounter] = pT*SfX/SfInlet/pDyn;
                }
                else if ( scale == -1.0 )
                {
                    objDiscreteVal[cfsurfCounter] = -pT*SfX/SfOutlet/pDyn;
                }
                else
                {
                    FatalErrorIn(" ")<<"userDefinedPatch scale not valid"<< abort(FatalError);   
                }
                cfsurfCounter++;
            }
        }
        else
        {
            label patchI = mesh_.boundaryMesh().findPatchID( patchName );
            forAll(mesh_.boundaryMesh()[patchI],faceI)
            {
                scalar pS = p.boundaryField()[patchI][faceI];
                scalar UMagIn = mag( U.boundaryField()[patchI][faceI] );
                scalar SfX = mesh_.magSf().boundaryField()[patchI][faceI];
                scalar rhoS = rho.boundaryField()[patchI][faceI];
                scalar pT = pS+0.5*rhoS*UMagIn*UMagIn;
                if(adjIO_.isInList<word>(patchName,adjIO_.inletPatches) )
                {
                    objDiscreteVal[cfsurfCounter] = pT*SfX/SfInlet/pDyn;
                }
                else if (adjIO_.isInList<word>(patchName,adjIO_.outletPatches) )
                {
                    objDiscreteVal[cfsurfCounter] = -pT*SfX/SfOutlet/pDyn;

                }
                else 
                {
                    FatalErrorIn(" ")<<"CPL Patch not valid"<< abort(FatalError);  
                }
                cfsurfCounter++;
            }
        }
    }
    
    objVal =0.0;
    forAll(objDiscreteVal,idxI)
    {
        objVal  += objDiscreteVal[idxI];
    }

    reduce(objVal , sumOp<scalar>() );

    return;
    
}

void AdjointObjectiveFunction::calcCD(scalar& objVal, scalarList& objDiscreteVal)
{
    vector fSum(this->calcForces());

    const scalar& magURef = adjIO_.referenceValues["magURef"];
    const scalar& rhoRef  = adjIO_.referenceValues["rhoRef"];
    const scalar& ARef  = adjIO_.referenceValues["ARef"];
    scalar pDyn = 0.5*rhoRef*magURef*magURef;

    // project forces to drag direction
    objVal = fSum & adjIO_.dragDir;
    // normalize drag
    objVal /= ARef*pDyn;

    if(adjIO_.solveAdjoint)
    {
        // project surfForces to drag direction
        forAll(surfForces_,idxI)
        {
            objDiscreteVal[idxI] = surfForces_[idxI] &adjIO_.dragDir;
            objDiscreteVal[idxI] /= ARef*pDyn;
        }
    }
}

void AdjointObjectiveFunction::calcCL(scalar& objVal, scalarList& objDiscreteVal)
{
    vector fSum(this->calcForces());

    const scalar& magURef = adjIO_.referenceValues["magURef"];
    const scalar& rhoRef  = adjIO_.referenceValues["rhoRef"];
    const scalar& ARef  = adjIO_.referenceValues["ARef"];
    scalar pDyn = 0.5*rhoRef*magURef*magURef;

    // project forces to drag direction
    objVal = fSum & adjIO_.liftDir;
    // normalize drag
    objVal /= ARef*pDyn;

    if(adjIO_.solveAdjoint)
    {
        // project surfForces to drag direction
        forAll(surfForces_,idxI)
        {
            objDiscreteVal[idxI] = surfForces_[idxI] &adjIO_.liftDir;
            objDiscreteVal[idxI] /= ARef*pDyn;
        }
    }
}

void AdjointObjectiveFunction::calcCMX(scalar& objVal, scalarList& objDiscreteVal)
{
    vector fSum(this->calcForces());

    const scalar& magURef = adjIO_.referenceValues["magURef"];
    const scalar& rhoRef  = adjIO_.referenceValues["rhoRef"];
    const scalar& ARef = adjIO_.referenceValues["ARef"];
    const scalar& LRef=adjIO_.referenceValues["LRef"];
    scalar pDyn = 0.5*rhoRef*magURef*magURef;

    scalar CMXSum = 0.0;
    label cfsurfCounter = 0;
    forAll(forcePatches_,cI)
    {
        // get the patch id label
        label patchI = mesh_.boundaryMesh().findPatchID( forcePatches_[cI] );
        // create a shorter handle for the boundary patch
        const fvPatch& patch = mesh_.boundary()[patchI];
        forAll(patch,faceI)
        {
            vector surfaceCenter = mesh_.Cf().boundaryField()[patchI][faceI] - adjIO_.CofR;
            vector surfaceTorque = surfaceCenter ^ surfForces_[cfsurfCounter];
            objDiscreteVal[cfsurfCounter] = surfaceTorque.x()/LRef/ARef/pDyn;
            CMXSum += objDiscreteVal[cfsurfCounter];
            cfsurfCounter+=1;
        }
    }

    reduce(CMXSum,sumOp<scalar>() );

    objVal = CMXSum;

    return;

}

void AdjointObjectiveFunction::calcCMY(scalar& objVal, scalarList& objDiscreteVal)
{
    vector fSum(this->calcForces());

    const scalar& magURef = adjIO_.referenceValues["magURef"];
    const scalar& rhoRef  = adjIO_.referenceValues["rhoRef"];
    const scalar& ARef  = adjIO_.referenceValues["ARef"];
    const scalar& LRef=adjIO_.referenceValues["LRef"];
    scalar pDyn = 0.5*rhoRef*magURef*magURef;

    scalar CMYSum = 0.0;
    label cfsurfCounter = 0;
    forAll(forcePatches_,cI)
    {
        // get the patch id label
        label patchI = mesh_.boundaryMesh().findPatchID( forcePatches_[cI] );
        // create a shorter handle for the boundary patch
        const fvPatch& patch = mesh_.boundary()[patchI];
        forAll(patch,faceI)
        {
            vector surfaceCenter = mesh_.Cf().boundaryField()[patchI][faceI] - adjIO_.CofR;
            vector surfaceTorque = surfaceCenter ^ surfForces_[cfsurfCounter];
            objDiscreteVal[cfsurfCounter] = surfaceTorque.y()/LRef/ARef/pDyn;
            CMYSum += objDiscreteVal[cfsurfCounter];
            cfsurfCounter+=1;
        }
    }
    
    reduce(CMYSum,sumOp<scalar>() );

    objVal = CMYSum;

    return;

}

void AdjointObjectiveFunction::calcCMZ(scalar& objVal, scalarList& objDiscreteVal)
{
    vector fSum(this->calcForces());

    const scalar& magURef = adjIO_.referenceValues["magURef"];
    const scalar& rhoRef  = adjIO_.referenceValues["rhoRef"];
    const scalar& ARef=adjIO_.referenceValues["ARef"];
    const scalar& LRef=adjIO_.referenceValues["LRef"];
    scalar pDyn = 0.5*rhoRef*magURef*magURef;

    scalar CMZSum = 0.0;
    label cfsurfCounter = 0;
    forAll(forcePatches_,cI)
    {
        // get the patch id label
        label patchI = mesh_.boundaryMesh().findPatchID( forcePatches_[cI] );
        // create a shorter handle for the boundary patch
        const fvPatch& patch = mesh_.boundary()[patchI];
        forAll(patch,faceI)
        {
            vector surfaceCenter = mesh_.Cf().boundaryField()[patchI][faceI] - adjIO_.CofR;
            vector surfaceTorque = surfaceCenter ^ surfForces_[cfsurfCounter];
            objDiscreteVal[cfsurfCounter] = surfaceTorque.z()/LRef/ARef/pDyn;
            CMZSum += objDiscreteVal[cfsurfCounter];
            cfsurfCounter+=1;
        }
    }

    reduce(CMZSum,sumOp<scalar>() );

    objVal = CMZSum;

    return;

}

void AdjointObjectiveFunction::calcAVGV(scalar& objVal, scalarList& objDiscreteVal, wordList geoNames)
{
    // Note : we are not taking abs value when averaging

    const objectRegistry& db = mesh_.thisDb();

    // compute the volume that is occupied by the prescibed userDefinedVolume
    label nGeoNames = geoNames.size();
    scalarList volSum(nGeoNames);
    forAll(volSum,idxI) volSum[idxI]=0.0;

    forAll(geoNames,idxI) 
    {
        word geoName=geoNames[idxI];
        forAll(adjIdx_.cellIdx4UserDefinedVolumes[geoName],idxJ)
        {
            label cellI = adjIdx_.cellIdx4UserDefinedVolumes[geoName][idxJ];
            volSum[idxI]+=mesh_.V()[cellI];
        }
        reduce(volSum[idxI], sumOp<scalar>() );
    }

    label counterI=0;
    forAll(geoNames,idxI)
    {
        word geoName=geoNames[idxI];

        dictionary geoDict = adjIO_.userDefinedVolumeInfo.subDict(geoName);
        word stateName = word(geoDict["stateName"]);
        scalar scale = readScalar( geoDict["scale"] );
        label comp = readLabel( geoDict["component"] );

        word stateType = adjIdx_.adjStateType[stateName];
    
        if (stateType=="volVectorState")
        {
            const volVectorField& state = db.lookupObject<volVectorField>(stateName);
            forAll(adjIdx_.cellIdx4UserDefinedVolumes[geoName],idxJ)
            {
                label cellI = adjIdx_.cellIdx4UserDefinedVolumes[geoName][idxJ];
                objDiscreteVal[counterI]=scale*state[cellI][comp]*mesh_.V()[cellI]/volSum[idxI];
                counterI++;
            }
        }
        else if (stateType=="volScalarState" or stateType=="turbState")
        {
            const volScalarField& state = db.lookupObject<volScalarField>(stateName);
            forAll(adjIdx_.cellIdx4UserDefinedVolumes[geoName],idxJ)
            {
                label cellI = adjIdx_.cellIdx4UserDefinedVolumes[geoName][idxJ];
                objDiscreteVal[counterI]=scale*state[cellI]*mesh_.V()[cellI]/volSum[idxI];
                counterI++;
            }
        }
        else
        {
            FatalErrorIn(" ")<<"stateID not supoorted!"<< abort(FatalError);
        }

    }

    objVal =0.0;
    forAll(objDiscreteVal,idxI)
    {
        objVal += objDiscreteVal[idxI];
    }

    reduce(objVal, sumOp<scalar>() );

    return;

}

void AdjointObjectiveFunction::calcVARV(scalar& objVal, scalarList& objDiscreteVal, wordList geoNames)
{
    // compute the variance of the prescribed variable in an user defined volume

    const objectRegistry& db = mesh_.thisDb();

    // first compute the volume that is occupied by the prescibed userDefinedVolume and reduce
    label nGeoNames = geoNames.size();
    scalarList volSum(nGeoNames);
    forAll(volSum,idxI) volSum[idxI]=0.0;

    forAll(geoNames,idxI) 
    {
        word geoName=geoNames[idxI];
        forAll(adjIdx_.cellIdx4UserDefinedVolumes[geoName],idxJ)
        {
            label cellI = adjIdx_.cellIdx4UserDefinedVolumes[geoName][idxJ];
            volSum[idxI]+=mesh_.V()[cellI];
        }
        reduce(volSum[idxI], sumOp<scalar>() );
    }

    // then we compute the average value that is occupied by the prescibed userDefinedVolume and reduce
    scalar objAvg = 0.0;
    forAll(geoNames,idxI) 
    {
        word geoName=geoNames[idxI];

        dictionary geoDict = adjIO_.userDefinedVolumeInfo.subDict(geoName);
        word stateName = word(geoDict["stateName"]);
        label comp = readLabel( geoDict["component"] );

        word stateType = adjIdx_.adjStateType[stateName];
    
        if (stateType=="volVectorState")
        {
            const volVectorField& state = db.lookupObject<volVectorField>(stateName);
            forAll(adjIdx_.cellIdx4UserDefinedVolumes[geoName],idxJ)
            {
                label cellI = adjIdx_.cellIdx4UserDefinedVolumes[geoName][idxJ];
                objAvg+= state[cellI][comp]*mesh_.V()[cellI]/volSum[idxI];
            }
        }
        else if (stateType=="volScalarState" or stateType=="turbState")
        {
            const volScalarField& state = db.lookupObject<volScalarField>(stateName);
            forAll(adjIdx_.cellIdx4UserDefinedVolumes[geoName],idxJ)
            {
                label cellI = adjIdx_.cellIdx4UserDefinedVolumes[geoName][idxJ];
                objAvg+= state[cellI]*mesh_.V()[cellI]/volSum[idxI];
            }
        }
        else
        {
            FatalErrorIn(" ")<<"stateID not supported!"<< abort(FatalError);
        }

    }
    reduce(objAvg, sumOp<scalar>() );

    // now we can repeat and compute variance
    label counterI=0;
    forAll(geoNames,idxI)
    {
        word geoName=geoNames[idxI];

        dictionary geoDict = adjIO_.userDefinedVolumeInfo.subDict(geoName);
        word stateName = word(geoDict["stateName"]);
        scalar scale = readScalar( geoDict["scale"] );
        label comp = readLabel( geoDict["component"] );

        word stateType = adjIdx_.adjStateType[stateName];
    
        if (stateType=="volVectorState")
        {
            const volVectorField& state = db.lookupObject<volVectorField>(stateName);
            forAll(adjIdx_.cellIdx4UserDefinedVolumes[geoName],idxJ)
            {
                label cellI = adjIdx_.cellIdx4UserDefinedVolumes[geoName][idxJ];
                objDiscreteVal[counterI]= (state[cellI][comp]-objAvg)
                                         *(state[cellI][comp]-objAvg)
                                         *mesh_.V()[cellI]*scale
                                         /volSum[idxI];
                counterI++;
            }
        }
        else if (stateType=="volScalarState" or stateType=="turbState")
        {
            const volScalarField& state = db.lookupObject<volScalarField>(stateName);
            forAll(adjIdx_.cellIdx4UserDefinedVolumes[geoName],idxJ)
            {
                label cellI = adjIdx_.cellIdx4UserDefinedVolumes[geoName][idxJ];
                objDiscreteVal[counterI]= (state[cellI]-objAvg)
                                         *(state[cellI]-objAvg)
                                         *mesh_.V()[cellI]*scale
                                         /volSum[idxI];
                counterI++;
            }
        }
        else
        {
            FatalErrorIn(" ")<<"stateID not supported!"<< abort(FatalError);
        }

    }

    objVal =0.0;
    forAll(objDiscreteVal,idxI)
    {
        objVal += objDiscreteVal[idxI];
    }

    reduce(objVal, sumOp<scalar>() );

    return;

}

void AdjointObjectiveFunction::calcAVGS(scalar& objVal, scalarList& objDiscreteVal, wordList geoNames)
{
    // Note : we are not taking abs value when averaging

    const objectRegistry& db = mesh_.thisDb();

    // compute the area that is occupied by the prescibed patches
    label nGeoNames = geoNames.size();
    scalarList areaSum(nGeoNames);
    forAll(areaSum,idxI) areaSum[idxI]=0.0;

    forAll(geoNames,idxI) 
    {
        word geoName=geoNames[idxI];
        if ( adjIdx_.isUserDefinedPatch(geoName))
        {
            forAll(adjIdx_.faceIdx4UserDefinedPatches[geoName],idxJ)
            {
                label faceI = adjIdx_.faceIdx4UserDefinedPatches[geoName][idxJ];
                if (faceI < adjIdx_.nLocalInternalFaces)
                {
                    areaSum[idxI]+= mesh_.magSf()[faceI];
                }
                else
                {
                    label relIdx=faceI-adjIdx_.nLocalInternalFaces;
                    label patchIdx=adjIdx_.bFacePatchI[relIdx];
                    label faceIdx=adjIdx_.bFaceFaceI[relIdx];
                    areaSum[idxI]+=mesh_.magSf().boundaryField()[patchIdx][faceIdx];
                }
            }
        }
        else
        {
            FatalErrorIn(" ")<<"AVGS only supports userDefinedPatch!"<< abort(FatalError);
        }

        reduce(areaSum[idxI], sumOp<scalar>() );
    }
    
    // now compute AVG
    label counterI=0;
    forAll(geoNames,idxI)
    {
        word geoName=geoNames[idxI];

        dictionary geoDict = adjIO_.userDefinedPatchInfo.subDict(geoName);
        word stateName = word(geoDict["stateName"]);
        scalar scale = readScalar(  geoDict["scale"] );
        label comp = readLabel( geoDict["component"] );
        word stateType = adjIdx_.adjStateType[stateName];
    
        if (stateType=="volVectorState")
        {
            const volVectorField& state = db.lookupObject<volVectorField>(stateName);
            surfaceVectorField stateInterp = fvc::interpolate(state);
            forAll(adjIdx_.faceIdx4UserDefinedPatches[geoName],idxJ)
            {
                label faceI = adjIdx_.faceIdx4UserDefinedPatches[geoName][idxJ];
                if (faceI<adjIdx_.nLocalInternalFaces)
                {
                    scalar areaD = mesh_.magSf()[faceI];
                    objDiscreteVal[counterI]=scale*stateInterp[faceI][comp]*areaD/areaSum[idxI];
                }
                else
                {
                    label relIdx=faceI-adjIdx_.nLocalInternalFaces;
                    label patchIdx=adjIdx_.bFacePatchI[relIdx];
                    label faceIdx=adjIdx_.bFaceFaceI[relIdx];
                    scalar valD=stateInterp.boundaryField()[patchIdx][faceIdx][comp];
                    scalar areaD = mesh_.magSf().boundaryField()[patchIdx][faceIdx];
                    objDiscreteVal[counterI]=scale*valD*areaD/areaSum[idxI];
                }
                counterI++;
            }
        }
        else if (stateType=="volScalarState" or stateType=="turbState")
        {
            const volScalarField& state = db.lookupObject<volScalarField>(stateName);
            forAll(adjIdx_.faceIdx4UserDefinedPatches[geoName],idxJ)
            {
                label faceI = adjIdx_.faceIdx4UserDefinedPatches[geoName][idxJ];
                if (faceI<adjIdx_.nLocalInternalFaces)
                {
                    scalar areaD = mesh_.magSf()[faceI];
                    surfaceScalarField stateInterp = fvc::interpolate(state);
                    objDiscreteVal[counterI]=scale*stateInterp[faceI]*areaD/areaSum[idxI];
                }
                else
                {
                    label relIdx=faceI-adjIdx_.nLocalInternalFaces;
                    label patchIdx=adjIdx_.bFacePatchI[relIdx];
                    label faceIdx=adjIdx_.bFaceFaceI[relIdx];
                    scalar valD=state.boundaryField()[patchIdx][faceIdx];
                    scalar areaD = mesh_.magSf().boundaryField()[patchIdx][faceIdx];
                    objDiscreteVal[counterI]=scale*valD*areaD/areaSum[idxI];
                }
                counterI++;
            }
        }
        else
        {
            FatalErrorIn(" ")<<"stateID not supoorted!"<< abort(FatalError);
        }

    }

    objVal =0.0;
    forAll(objDiscreteVal,idxI)
    {
        objVal += objDiscreteVal[idxI];
    }

    reduce(objVal, sumOp<scalar>() );

    return;

}

// for immersed boundary propeller
void AdjointObjectiveFunction::calcActuatorDiskSource(vectorField& sourceU)
{
    // Reference: 
    // Hoekstra, A RANS-based analysis tool for ducted propeller systems in open water condition, International Shipbuilding Progress

    const scalar& magURef=adjIO_.referenceValues["magURef"];
    const scalar& rhoRef=adjIO_.referenceValues["rhoRef"];
    const scalar& ARef=adjIO_.referenceValues["ARef"];
    const scalar& LRef=adjIO_.referenceValues["LRef"];
    scalar pDyn = 0.5*rhoRef*magURef*magURef;
    
    forAll(adjIO_.actuatorVolumeNames,idxI)
    {
        // first read the parameters from userDefinedVolumeInfo
        word geoName=adjIO_.actuatorVolumeNames[idxI];

        dictionary geoDict = adjIO_.userDefinedVolumeInfo.subDict(geoName);
        word geoType = word(geoDict["type"]);
        if (geoType!="annulus") FatalErrorIn(" ")<<"only support geoType=annulus  volume!"<< abort(FatalError);   

        vector actuatorCenter(vector::zero);
        actuatorCenter[0]= readScalar( geoDict["centerX"] );
        actuatorCenter[1]= readScalar( geoDict["centerY"] );
        actuatorCenter[2]= readScalar( geoDict["centerZ"] );
        scalar actuatorHubR = readScalar( geoDict["radiusInner"] ); // inner radius
        scalar actuatorTipR = readScalar( geoDict["radiusOuter"] ); // outer radius
        word axis    = word(geoDict["axis"]);
        vector actuatorDir(vector::zero);
        if(axis=="x")
        {
            actuatorDir={1.0,0.0,0.0};
        }
        else if (axis=="y")
        {
            actuatorDir={0.0,1.0,0.0};
        }
        else if (axis=="z")
        {
            actuatorDir={0.0,0.0,1.0};
        }
        else
        {
            FatalErrorIn(" ")<<"axis not valid!"<< abort(FatalError);  
        }

        // now compute stuff
        vector actuatorDirNorm(vector::zero);
    
        actuatorDirNorm = actuatorDir / mag(actuatorDir);
        
        scalar targetThrust = 0.0;
        if (adjIO_.actuatorAdjustThrust)
        {
            vector fSum(this->calcForces());
            targetThrust = fSum&actuatorDir;
        }
        else
        {
            targetThrust = adjIO_.actuatorThrustCoeff[idxI]*ARef*pDyn;
        }
    
        Info<<"ThrustCoeff Target for "<<geoName<<": "<<targetThrust/ARef/pDyn<<endl;
            
        scalar myPI = 3.14159265358979;
        
        // first calculate a scaling factor to make sure all the source term sums up to Thrust
        scalar scaleC = 0.0;
        //scalar acutatorDiskVol = 0.0;
        forAll(adjIdx_.cellIdx4UserDefinedVolumes[geoName],idxJ)
        {
            label cellI = adjIdx_.cellIdx4UserDefinedVolumes[geoName][idxJ];

            vector cellC = mesh_.C()[cellI];
            vector cellC2AVec = cellC-actuatorCenter; // cell center to disk center vector
            
            tensor cellC2AVecE(tensor::zero); // tmp tensor for calculating the axial/radial components of cellC2AVec
            cellC2AVecE.xx() = cellC2AVec.x();
            cellC2AVecE.yy() = cellC2AVec.y();
            cellC2AVecE.zz() = cellC2AVec.z();
            
            vector cellC2AVecA = cellC2AVecE & actuatorDirNorm; // axial
            vector cellC2AVecR = cellC2AVec-cellC2AVecA; // radial
    
            //scalar cellC2AVecALen =  mag(cellC2AVecA);
            scalar cellC2AVecRLen =  mag(cellC2AVecR);

            scalar rPrime =  cellC2AVecRLen / actuatorTipR;
            scalar rPrimeHub = actuatorHubR / actuatorTipR;
            scalar rStar  = (rPrime - rPrimeHub)/(1.0-rPrimeHub);
            scaleC += rStar*Foam::sqrt(1.0-rStar);
        }
        reduce(scaleC,sumOp<scalar>());

        // now we repeat the loop for the source term
        scalar thrustSourceSum = 0.0;
        scalar torqueSourceSum = 0.0;
        forAll(adjIdx_.cellIdx4UserDefinedVolumes[geoName],idxJ)
        {
            label cellI = adjIdx_.cellIdx4UserDefinedVolumes[geoName][idxJ];

            vector cellC = mesh_.C()[cellI];
            vector cellC2AVec = cellC-actuatorCenter; // cell center to disk center vector
            
            tensor cellC2AVecE(tensor::zero); // tmp tensor for calculating the axial/radial components of cellC2AVec
            cellC2AVecE.xx() = cellC2AVec.x();
            cellC2AVecE.yy() = cellC2AVec.y();
            cellC2AVecE.zz() = cellC2AVec.z();
            
            vector cellC2AVecA = cellC2AVecE & actuatorDirNorm; // axial
            vector cellC2AVecR = cellC2AVec-cellC2AVecA; // radial

            vector cellC2AVecC(vector::zero);
            if(adjIO_.actuatorRotationDir[idxI]=="left")
            {
                 // this assumes right hand rotation of propellers
                cellC2AVecC = cellC2AVecR ^ cellC2AVecA; // circ
            }
            else if (adjIO_.actuatorRotationDir[idxI]=="right")
            {
                // this assumes left hand rotation of propellers
                cellC2AVecC = cellC2AVecA ^ cellC2AVecR; // circ
            }
            else
            {
                FatalErrorIn(" ")<<"rotation dir not valid"<< abort(FatalError); 
            }
    
            //scalar cellC2AVecALen =  mag(cellC2AVecA);
            scalar cellC2AVecRLen =  mag(cellC2AVecR);
            scalar cellC2AVecCLen =  mag(cellC2AVecC);
            
            //vector cellC2AVecANorm = cellC2AVecA/ cellC2AVecALen;
            vector cellC2AVecCNorm = cellC2AVecC/ cellC2AVecCLen;
        
            scalar rPrime =  cellC2AVecRLen / actuatorTipR;
            scalar rPrimeHub = actuatorHubR / actuatorTipR;
            scalar rStar  = (rPrime - rPrimeHub)/(1.0-rPrimeHub);
            scalar fAxial = targetThrust*rStar*Foam::sqrt(1.0-rStar)/scaleC;
            // we use Hoekstra's method to calculate the fCirc based on fAxial
            scalar fCirc  = fAxial* adjIO_.actuatorPOverD[idxI] / myPI / rPrime;
            vector sourceVec = (fAxial*actuatorDirNorm+fCirc*cellC2AVecCNorm);
            sourceU[cellI] += sourceVec;
            thrustSourceSum += fAxial;
            torqueSourceSum += fCirc;
            
        }
    
        reduce(thrustSourceSum,sumOp<scalar>());
        reduce(torqueSourceSum,sumOp<scalar>());
        
        Info<<"ThrustCoeff Source Term for "<<geoName<<": "<<thrustSourceSum/ARef/pDyn<<endl;
        Info<<"TorqueCoeff Source Term for "<<geoName<<": "<<torqueSourceSum/ARef/pDyn/LRef<<endl;

    }
    
    return;

}

void AdjointObjectiveFunction::calcVMS(scalar& objVal, scalarList& objDiscreteVal,scalar& KSExpSum)
{ 
    // calculate the KS aggregated von Mises stress
    // VMS = KS( mu*twoSymm(fvc::grad(D)) + lambda*(I*tr(fvc::grad(D))) )
    // KS function KS(x) = 1/KSCoeff * ln( sum[exp(KSCoeff*x_i)] ) 

    // ********************* NOTE **************
    // To enable coloring for the KS function, discreteVal stores only exp(KSCoeff*x_i)
    // We will do a scaling in the calcdFdW fuction to take account of the 1/KSCoeff*ln(.) part
    // and get the right derivatives

    forAll(objDiscreteVal,idxI) objDiscreteVal[idxI]=0.0;
    
    const objectRegistry& db = mesh_.thisDb();
    const volVectorField& D = db.lookupObject<volVectorField>("D");
    const volScalarField& lambda = db.lookupObject<volScalarField>("solid:lambda");
    const volScalarField& mu = db.lookupObject<volScalarField>("solid:mu");
    const volScalarField& rho = db.lookupObject<volScalarField>("solid:rho");

    volSymmTensorField sigma = rho*( mu*twoSymm(fvc::grad(D)) + lambda*(I*tr(fvc::grad(D))) );

    if (thermalStress)
    {
        const volScalarField& T = db.lookupObject<volScalarField>("T");
        const volScalarField& threeKalpha = db.lookupObject<volScalarField>("threeKalpha");
        sigma = sigma - I*(rho*threeKalpha*T);
    }

    volScalarField vonMises = sqrt((3.0/2.0)*magSqr(dev(sigma)));

    scalar KSCoeff =adjIO_.referenceValues["KSCoeff"];

    // now do KS function of vonMises
    scalar objValTmp=0.0;
    forAll(mesh_.cells(),cellI)
    {
        objDiscreteVal[cellI]=exp(KSCoeff*vonMises[cellI]);
        objValTmp += objDiscreteVal[cellI];
        if (objValTmp > 1e200) 
        {
            FatalErrorIn(" ")<<"KS function summation term too large! "
                             <<"Reduce KSCoeff! "<< abort(FatalError);   
        }
    }

    reduce(objValTmp, sumOp<scalar>() );

    // KSExpSum stores sum[exp(KSCoeff*x_i)], it will be used to scale dFdW
    // in the calcdFdW function
    KSExpSum=objValTmp; 

    objVal = log(objValTmp)/KSCoeff;
    
    return;
    
}

void AdjointObjectiveFunction::calcTPR(scalar& objVal, scalarList& objDiscreteVal, label isRef)
{
    // calculate total pressure ratio,  TPIn/TPOut
    // NOTE: to enable coloring, we need to separate TPIn and TPOut for each face
    // to this end, we do:
    // d(TPInAvg/TPOutAvg)/dw = -TPOutAvg/TPInAvg^2 * dTPInAvg/dw + 1/TPInAvg * dTPOutAvg/dw
    // so here objDiscreteVal = -TPOutAvg/TPInAvg^2 * dTPIn_i*ds_i/dSIn for inlet patches
    // and objDiscreteVal = 1/TPInAvg * TPOut_i*ds_i/dSOut for outlet patches

    const scalar& Cp=adjIO_.flowProperties["Cp"];
    const scalar gamma = 1.4;
    const scalar R = Cp-Cp/gamma;
    const scalar expCoeff = gamma/(gamma-1.0);

    wordList tprPatches;
    forAll(objFuncs_,idxI)
    {
        if(objFuncs_[idxI]=="TPR") tprPatches=objFuncGeoInfo_[idxI];
    }

    const objectRegistry& db = mesh_.thisDb();
    const volScalarField& p = db.lookupObject<volScalarField>("p");
    const volScalarField& T = db.lookupObject<volScalarField>("T");
    const volVectorField& U = db.lookupObject<volVectorField>("U");

    // calc the inlet and outlet area
    scalar SfInlet=0.0;
    scalar SfOutlet=0.0;
    forAll(tprPatches,idxI)
    {
        word patchName = tprPatches[idxI];
        label patchI = mesh_.boundaryMesh().findPatchID( patchName );
        forAll(mesh_.boundaryMesh()[patchI],faceI)
        {
            if(adjIO_.isInList<word>(patchName,adjIO_.inletPatches) )
            {
                SfInlet += mesh_.magSf().boundaryField()[patchI][faceI];
            }
            else if (adjIO_.isInList<word>(patchName,adjIO_.outletPatches) )
            {
                SfOutlet += mesh_.magSf().boundaryField()[patchI][faceI];
            }
            else
            {
                FatalErrorIn(" ")<<"TPR patch not valid"<< abort(FatalError);   
            }
        }
    }
    reduce(SfInlet, sumOp<scalar>() );
    reduce(SfOutlet, sumOp<scalar>() );

    // we first compute the averaged inlet and outlet total pressure they will 
    // be used later for normalization
    scalar TPIn=0.;
    scalar TPOut=0.;
    forAll(tprPatches,idxI)
    {
        word patchName = tprPatches[idxI];
        label patchI = mesh_.boundaryMesh().findPatchID( patchName );
        forAll(mesh_.boundaryMesh()[patchI],faceI)
        {
            scalar pS = p.boundaryField()[patchI][faceI];
            scalar TS = T.boundaryField()[patchI][faceI];
            scalar UMag = mag( U.boundaryField()[patchI][faceI] );
            scalar SfX = mesh_.magSf().boundaryField()[patchI][faceI];
            scalar Ma2 = Foam::sqr( UMag/Foam::sqrt(gamma*R*TS) );
            scalar pT = pS*Foam::pow(1.0+0.5*(gamma - 1.0)*Ma2, expCoeff);

            if(adjIO_.isInList<word>(patchName,adjIO_.inletPatches) )
            {
                TPIn+=pT*SfX/SfInlet;
            }
            else if (adjIO_.isInList<word>(patchName,adjIO_.outletPatches) )
            {
                TPOut+=pT*SfX/SfOutlet;
            }
        }
    }
    reduce(TPIn, sumOp<scalar>() );
    reduce(TPOut, sumOp<scalar>() );

    // set Reference values
    if(isRef)
    {
        TPInRef_ = TPIn;
        TPOutRef_ = TPOut;
    }
    else
    {
        // this is for running flow only
        if(fabs(TPInRef_)<1e-10 || fabs(TPOutRef_)<1e-10) 
        {
            TPInRef_ = TPIn;
            TPOutRef_ = TPOut;
        }
    }

    // pTotal ratio: Note we need special treatment to enable coloring
    label cfsurfCounter = 0;
    forAll(tprPatches,idxI)
    {
        word patchName = tprPatches[idxI];
        label patchI = mesh_.boundaryMesh().findPatchID( patchName );
        forAll(mesh_.boundaryMesh()[patchI],faceI)
        {
            scalar pS = p.boundaryField()[patchI][faceI];
            scalar TS = T.boundaryField()[patchI][faceI];
            scalar UMag = mag( U.boundaryField()[patchI][faceI] );
            scalar SfX = mesh_.magSf().boundaryField()[patchI][faceI];
            scalar Ma2 = Foam::sqr( UMag/Foam::sqrt(gamma*R*TS) );
            scalar pT = pS*Foam::pow(1.0+0.5*(gamma - 1.0)*Ma2, expCoeff);

            if(adjIO_.isInList<word>(patchName,adjIO_.inletPatches) )
            {
                objDiscreteVal[cfsurfCounter] = -pT*SfX/SfInlet*TPOutRef_/TPInRef_/TPInRef_;
            }
            else if (adjIO_.isInList<word>(patchName,adjIO_.outletPatches) )
            {
                objDiscreteVal[cfsurfCounter] = pT*SfX/SfOutlet/TPInRef_;
            }
            cfsurfCounter++;
        }
    }
    
    objVal =TPOut/TPIn;

    return;
    
}

void AdjointObjectiveFunction::calcMFR(scalar& objVal, scalarList& objDiscreteVal)
{
    // calculate mass flow rate

    wordList mfrPatches;
    forAll(objFuncs_,idxI)
    {
        if(objFuncs_[idxI]=="MFR") mfrPatches=objFuncGeoInfo_[idxI];
    }
        
    const objectRegistry& db = mesh_.thisDb();
    volScalarField rho  = adjRAS_.getRho();
    const volVectorField& U = db.lookupObject<volVectorField>("U");

    // calc the area
    scalar SfTotal=0.0;
    forAll(mfrPatches,idxI)
    {
        word patchName = mfrPatches[idxI];
        label patchI = mesh_.boundaryMesh().findPatchID( patchName );
        forAll(mesh_.boundaryMesh()[patchI],faceI)
        {
            SfTotal += mesh_.magSf().boundaryField()[patchI][faceI];
        }
    }
    reduce(SfTotal, sumOp<scalar>() );

    // mass flow rate
    label cfsurfCounter = 0;
    forAll(mfrPatches,idxI)
    {
        word patchName = mfrPatches[idxI];
        label patchI = mesh_.boundaryMesh().findPatchID( patchName );
        forAll(mesh_.boundaryMesh()[patchI],faceI)
        {
            
            vector US = U.boundaryField()[patchI][faceI];
            vector Sf = mesh_.Sf().boundaryField()[patchI][faceI];
            scalar rhoS = rho.boundaryField()[patchI][faceI];
            scalar mfr = rhoS*(US & Sf);
            objDiscreteVal[cfsurfCounter] = mfr;
            cfsurfCounter++;
        }
    }
    
    objVal =0.0;
    forAll(objDiscreteVal,idxI)
    {
        objVal  += objDiscreteVal[idxI];
    }

    reduce(objVal , sumOp<scalar>() );

    return;
    
}

void AdjointObjectiveFunction::calcTTR(scalar& objVal, scalarList& objDiscreteVal, label isRef)
{
    // calculate total temperature ratio,  TTIn/TTOut
    // NOTE: to enable coloring, we need to separate TTIn and TTOut for each face
    // to this end, we do:
    // d(TTInAvg/TTOutAvg)/dw = -TTOutAvg/TTInAvg^2 * dTTInAvg/dw + 1/TTInAvg * dTTOutAvg/dw
    // so here objDiscreteVal = -TTOutAvg/TTInAvg^2 * dTTIn_i*ds_i/dSIn for inlet patches
    // and objDiscreteVal = 1/TTInAvg * TTOut_i*ds_i/dSOut for outlet patches

    const scalar& Cp=adjIO_.flowProperties["Cp"];
    const scalar gamma = 1.4;
    const scalar R = Cp-Cp/gamma;

    wordList ttrPatches;
    forAll(objFuncs_,idxI)
    {
        if(objFuncs_[idxI]=="TTR") ttrPatches=objFuncGeoInfo_[idxI];
    }

    const objectRegistry& db = mesh_.thisDb();
    const volScalarField& T = db.lookupObject<volScalarField>("T");
    const volVectorField& U = db.lookupObject<volVectorField>("U");

    // calc the inlet and outlet area
    scalar SfInlet=0.0;
    scalar SfOutlet=0.0;
    forAll(ttrPatches,idxI)
    {
        word patchName = ttrPatches[idxI];
        label patchI = mesh_.boundaryMesh().findPatchID( patchName );
        forAll(mesh_.boundaryMesh()[patchI],faceI)
        {
            if(adjIO_.isInList<word>(patchName,adjIO_.inletPatches) )
            {
                SfInlet += mesh_.magSf().boundaryField()[patchI][faceI];
            }
            else if (adjIO_.isInList<word>(patchName,adjIO_.outletPatches) )
            {
                SfOutlet += mesh_.magSf().boundaryField()[patchI][faceI];
            }
            else
            {
                FatalErrorIn(" ")<<"TTR patch not valid"<< abort(FatalError);   
            }
        }
    }
    reduce(SfInlet, sumOp<scalar>() );
    reduce(SfOutlet, sumOp<scalar>() );

    // we first compute the averaged inlet and outlet total temperature they will 
    // be used later for normalization
    scalar TTIn=0.;
    scalar TTOut=0.;
    forAll(ttrPatches,idxI)
    {
        word patchName = ttrPatches[idxI];
        label patchI = mesh_.boundaryMesh().findPatchID( patchName );
        forAll(mesh_.boundaryMesh()[patchI],faceI)
        {
            scalar TS = T.boundaryField()[patchI][faceI];
            scalar UMag = mag( U.boundaryField()[patchI][faceI] );
            scalar SfX = mesh_.magSf().boundaryField()[patchI][faceI];
            scalar Ma2 = Foam::sqr( UMag/Foam::sqrt(gamma*R*TS) );
            scalar TT = TS*(1.0+0.5*(gamma - 1.0)*Ma2);

            if(adjIO_.isInList<word>(patchName,adjIO_.inletPatches) )
            {
                TTIn+=TT*SfX/SfInlet;
            }
            else if (adjIO_.isInList<word>(patchName,adjIO_.outletPatches) )
            {
                TTOut+=TT*SfX/SfOutlet;
            }
        }
    }
    reduce(TTIn, sumOp<scalar>() );
    reduce(TTOut, sumOp<scalar>() );

    // set Reference values
    if(isRef)
    {
        TTInRef_ = TTIn;
        TTOutRef_ = TTOut;
    }
    else
    {
        if(TTInRef_ == 0 || TTOutRef_==0) 
        {
            TTInRef_ = TTIn;
            TTOutRef_ = TTOut;
        }
    }

    // TT ratio: Note we need special treatment to enable coloring
    label cfsurfCounter = 0;
    forAll(ttrPatches,idxI)
    {
        word patchName = ttrPatches[idxI];
        label patchI = mesh_.boundaryMesh().findPatchID( patchName );
        forAll(mesh_.boundaryMesh()[patchI],faceI)
        {
            scalar TS = T.boundaryField()[patchI][faceI];
            scalar UMag = mag( U.boundaryField()[patchI][faceI] );
            scalar SfX = mesh_.magSf().boundaryField()[patchI][faceI];
            scalar Ma2 = Foam::sqr( UMag/Foam::sqrt(gamma*R*TS) );
            scalar TT = TS*(1.0+0.5*(gamma - 1.0)*Ma2);

            if(adjIO_.isInList<word>(patchName,adjIO_.inletPatches) )
            {
                objDiscreteVal[cfsurfCounter] = -TT*SfX/SfInlet*TTOutRef_/TTInRef_/TTInRef_;
            }
            else if (adjIO_.isInList<word>(patchName,adjIO_.outletPatches) )
            {
                objDiscreteVal[cfsurfCounter] = TT*SfX/SfOutlet/TTInRef_;
            }
            cfsurfCounter++;
        }
    }
    
    objVal = TTOut/TTIn;


    return;
    
}

void AdjointObjectiveFunction::calcObjFuncs(word objFunc, label isRef)
{
    label objFuncID = objFuncsRegID_[objFunc];
    
    if (objFunc =="CD")
    {
        if(isRef)
        {
            this->calcCD(objFuncsValRef_[objFuncID],objFuncsDiscreteValRef_[objFuncID]);
        }
        else
        {
            this->calcCD(objFuncsVal_[objFuncID],objFuncsDiscreteVal_[objFuncID]);
        }
    }
    else if (objFunc == "CL")
    {
        if(isRef)
        {
            this->calcCL(objFuncsValRef_[objFuncID],objFuncsDiscreteValRef_[objFuncID]);
        }
        else
        {
            this->calcCL(objFuncsVal_[objFuncID],objFuncsDiscreteVal_[objFuncID]);
        }
    }
    else if (objFunc == "CMX")
    {
        if(isRef)
        {
            this->calcCMX(objFuncsValRef_[objFuncID],objFuncsDiscreteValRef_[objFuncID]);
        }
        else
        {
            this->calcCMX(objFuncsVal_[objFuncID],objFuncsDiscreteVal_[objFuncID]);
        }
    }
    else if (objFunc == "CMY")
    {
        if(isRef)
        {
            this->calcCMY(objFuncsValRef_[objFuncID],objFuncsDiscreteValRef_[objFuncID]);
        }
        else
        {
            this->calcCMY(objFuncsVal_[objFuncID],objFuncsDiscreteVal_[objFuncID]);
        }
    }
    else if (objFunc == "CMZ")
    {
        if(isRef)
        {
            this->calcCMZ(objFuncsValRef_[objFuncID],objFuncsDiscreteValRef_[objFuncID]);
        }
        else
        {
            this->calcCMZ(objFuncsVal_[objFuncID],objFuncsDiscreteVal_[objFuncID]);
        }
    }
    else if (objFunc == "CPL")
    {
        if(isRef)
        {
            this->calcCPL(objFuncsValRef_[objFuncID],objFuncsDiscreteValRef_[objFuncID]);
        }
        else
        {
            this->calcCPL(objFuncsVal_[objFuncID],objFuncsDiscreteVal_[objFuncID]);
        }
    }
    else if (objFunc == "NUS")
    {
        if(isRef)
        {
            this->calcNUS(objFuncsValRef_[objFuncID],objFuncsDiscreteValRef_[objFuncID]);
        }
        else
        {
            this->calcNUS(objFuncsVal_[objFuncID],objFuncsDiscreteVal_[objFuncID]);
        }
        
    }
    else if (objFunc == "AVGV")
    {
        wordList geoNames = adjIdx_.getObjFuncGeoInfo(objFunc);  
        if(isRef)
        {
            this->calcAVGV(objFuncsValRef_[objFuncID],objFuncsDiscreteValRef_[objFuncID],geoNames);
        }
        else
        {
            this->calcAVGV(objFuncsVal_[objFuncID],objFuncsDiscreteVal_[objFuncID],geoNames);
        }
        
    }
    else if (objFunc == "VARV")
    {
        wordList geoNames = adjIdx_.getObjFuncGeoInfo(objFunc);  
        if(isRef)
        {
            this->calcVARV(objFuncsValRef_[objFuncID],objFuncsDiscreteValRef_[objFuncID],geoNames);
        }
        else
        {
            this->calcVARV(objFuncsVal_[objFuncID],objFuncsDiscreteVal_[objFuncID],geoNames);
        }
        
    }
    else if (objFunc == "AVGS")
    {
        wordList geoNames = adjIdx_.getObjFuncGeoInfo(objFunc);  
        if(isRef)
        {
            this->calcAVGS(objFuncsValRef_[objFuncID],objFuncsDiscreteValRef_[objFuncID],geoNames);
        }
        else
        {
            this->calcAVGS(objFuncsVal_[objFuncID],objFuncsDiscreteVal_[objFuncID],geoNames);
        }
        
    }
    else if (objFunc == "VMS")
    {
        if(isRef)
        {
            this->calcVMS(objFuncsValRef_[objFuncID],objFuncsDiscreteValRef_[objFuncID],KSExpSumRef);
        }
        else
        {
            scalar tmp;
            this->calcVMS(objFuncsVal_[objFuncID],objFuncsDiscreteVal_[objFuncID],tmp);
        }
        
    }
    else if (objFunc == "TPR")
    {
        wordList geoNames = adjIdx_.getObjFuncGeoInfo(objFunc);  
        if(isRef)
        {
            this->calcTPR(objFuncsValRef_[objFuncID],objFuncsDiscreteValRef_[objFuncID],isRef);
        }
        else
        {
            this->calcTPR(objFuncsVal_[objFuncID],objFuncsDiscreteVal_[objFuncID],isRef);
        }
        
    }
    else if (objFunc == "TTR")
    {
        wordList geoNames = adjIdx_.getObjFuncGeoInfo(objFunc);  
        if(isRef)
        {
            this->calcTTR(objFuncsValRef_[objFuncID],objFuncsDiscreteValRef_[objFuncID],isRef);
        }
        else
        {
            this->calcTTR(objFuncsVal_[objFuncID],objFuncsDiscreteVal_[objFuncID],isRef);
        }
        
    }
    else if (objFunc == "MFR")
    {
        wordList geoNames = adjIdx_.getObjFuncGeoInfo(objFunc);  
        if(isRef)
        {
            this->calcMFR(objFuncsValRef_[objFuncID],objFuncsDiscreteValRef_[objFuncID]);
        }
        else
        {
            this->calcMFR(objFuncsVal_[objFuncID],objFuncsDiscreteVal_[objFuncID]);
        }
        
    }
    else
    {
        FatalErrorIn(" ")<<"objFunc not valid"<< abort(FatalError);   
    }

    return;

}

void AdjointObjectiveFunction::calcObjFuncPartDerivs(const scalar eps, const word objFunc)
{

    label isRef=0;
    this->calcObjFuncs(objFunc,isRef);

    label objFuncID = objFuncsRegID_[objFunc];

    objFuncsPartialDeriv_[objFuncID] = ( objFuncsVal_[objFuncID] 
                                       - objFuncsValRef_[objFuncID] ) / eps;
    objFuncsDiscretePartialDeriv_[objFuncID] = ( objFuncsDiscreteVal_[objFuncID] 
                                           - objFuncsDiscreteValRef_[objFuncID] ) / eps;
}

void AdjointObjectiveFunction::initializeSurfForces()
{
    // if surfForces has been initialized, return;
    if (surfForces_.size()!=0) 
    {
        return;
    }
         
    // First check if the objFuncPatches are identical for all force-related objFunc
    // in the objFuncs list. This is to avoid: e.g. setting different patches for CD and CL
    if (objFuncs_.size()>1)
    {

        List<word> refPatches(0);
        label refSet=0;
        forAll(objFuncs_,idxI)
        {
            if ( adjIO_.isInList<word>(objFuncs_[idxI],forceRelatedObjFuncs_) && refSet==0 )
            {
                refPatches=objFuncGeoInfo_[idxI];
                refSet=1;
                continue;
            }
            
            if ( adjIO_.isInList<word>(objFuncs_[idxI],forceRelatedObjFuncs_) && refSet==1 )
            {
                if ( objFuncGeoInfo_[idxI]!=refPatches)
                {
                    FatalErrorIn("")<< "The force-related objFuncPathces are not identical!"
                    <<abort(FatalError);
                }
            }
        }
    }

    // if any of the force related obj funcs are set, initialize surfForces_ vector   
    forAll(objFuncs_,idxI)
    {
        if( adjIO_.isInList<word>(objFuncs_[idxI],forceRelatedObjFuncs_) )
        {
            label nFaces = adjIdx_.getNLocalObjFuncGeoElements(objFuncs_[idxI]);
            surfForces_.setSize(nFaces);
            return;
        }
    }
    
    // if none of the above obj funcs are set, return and do nothing
    return;
}

void AdjointObjectiveFunction::initializeForcePatches()
{

    // if any of the force related obj funcs are set, get their patches    
    forAll(objFuncs_,idxI)
    {
        if( adjIO_.isInList<word>(objFuncs_[idxI],forceRelatedObjFuncs_) )
        {
            forcePatches_ = objFuncGeoInfo_[idxI];
            return;
        }
    }
    
    // if no force related obj functions are set just return
    
    return;
}

void AdjointObjectiveFunction::printObjFuncValues()
{
    forAll(objFuncs_,idxI)
    {
        word objFunc = objFuncs_[idxI];
        label objFuncID = objFuncsRegID_[objFunc];
        label isRef=0;
        this->calcObjFuncs(objFunc,isRef);
        Info<<objFunc<<": "<<objFuncsVal_[objFuncID]<<endl;
    }

    return;
}

void AdjointObjectiveFunction::writeObjFuncValues()
{
    Info<<"Writing Objective Function Values to objFuncs.dat"<<endl;

    OFstream fOut("objFuncs.dat");

    forAll(objFuncs_,idxI)
    {
        word objFunc = objFuncs_[idxI];
        label objFuncID = objFuncsRegID_[objFunc];
        label isRef=0;
        this->calcObjFuncs(objFunc,isRef);
        Info<<objFunc<<": "<<objFuncsVal_[objFuncID]<<endl;
        fOut<<objFunc<<" "<<objFuncsVal_[objFuncID]<<endl;
        
        if(objFunc=="NUS") 
        {
            NUSField_.write();
            TBulk_.write();
        }
    }
    
    return;
}

scalar AdjointObjectiveFunction::getObjFuncPartDeriv(const word objFunc)
{
    label objFuncID = objFuncsRegID_[objFunc];
    return objFuncsPartialDeriv_[objFuncID];
}

scalar AdjointObjectiveFunction::getObjFuncDiscretePartDeriv(word objFunc,label idxI)
{
    label objFuncID = objFuncsRegID_[objFunc];
    return objFuncsDiscretePartialDeriv_[objFuncID][idxI];
}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
