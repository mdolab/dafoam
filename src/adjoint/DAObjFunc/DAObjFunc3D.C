/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v3

\*---------------------------------------------------------------------------*/

#include "DAObjFunc3D.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

using namespace std;
namespace Foam

{

defineTypeNameAndDebug(DAObjFunc3D, 0);
addToRunTimeSelectionTable(DAObjFunc, DAObjFunc3D, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DAObjFunc3D::DAObjFunc3D(
    const fvMesh& mesh,
    const DAOption& daOption,
    const DAModel& daModel,
    const DAIndex& daIndex,
    const DAResidual& daResidual,
    const word objFuncName,
    const word objFuncPart,
    const dictionary& objFuncDict)
    : DAObjFunc(
        mesh,
        daOption,
        daModel,
        daIndex,
        daResidual,
        objFuncName,
        objFuncPart,
        objFuncDict),
        ubDataPtr_(nullptr),
      daTurb_(daModel.getDATurbulenceModel())

{
    /*
    Description:
        Calculate the stateErrorNorm
        f = scale * L2Norm( state-stateRef )*V/N
    Input:
        objFuncFaceSources: List of face source (index) for this objective
    
        objFuncCellSources: List of cell source (index) for this objective
    Output:
        objFuncFaceValues: the discrete value of objective for each face source (index). 
        This  will be used for computing df/dw in the adjoint.
    
        objFuncCellValues: the discrete value of objective on each cell source (index). 
        This will be used for computing df/dw in the adjoint.
    
        objFuncValue: the sum of objective, reduced across all processors and scaled by "scale"
    */

    objFuncDict_.readEntry<word>("type", objFuncType_);
    data_ = objFuncDict_.getWord("data");
    scale_ = objFuncDict_.getScalar("scale");
    objFuncDict_.readEntry<bool>("weightedSum", weightedSum_);
    if (weightedSum_ == true)
    {
        objFuncDict_.readEntry<scalar>("weight", weight_);
    }
    
    // setup the connectivity, this is needed in Foam::DAJacCondFdW
    // this objFunc only depends on the state variable at the zero level cell
    if (DAUtility::isInList<word>(stateName_, daIndex.adjStateNames))
    {
        objFuncConInfo_ = {{stateName_}}; // level 0
    }
    else
    {
        objFuncConInfo_ = {{}}; // level 0
    }
    if (daOption.getOption<word>("use3D") == "yes")
    {
        fvMesh& target = targetPtr_();
        Info << "Reading field uData\n"
                << endl;
        ubDataPtr_.reset(
                new volVectorField(
                IOobject(
                "ubData",
                target.time().timeName(),
                target,
                IOobject::MUST_READ,
                IOobject::AUTO_WRITE),
                target
                )
            );
    }

}

/// calculate the value of objective function
void DAObjFunc3D::calcObjFunc(
    const labelList& objFuncFaceSources,
    const labelList& objFuncCellSources,
    scalarList& objFuncFaceValues,
    scalarList& objFuncCellValues,
    scalar& objFuncValue)
{	
    /*
    Description:
        Calculate the stateErrorNorm
        f = scale * L2Norm( state-stateRef )

    Input:
        objFuncFaceSources: List of face source (index) for this objective
    
        objFuncCellSources: List of cell source (index) for this objective

    Output:
        objFuncFaceValues: the discrete value of objective for each face source (index). 
        This  will be used for computing df/dw in the adjoint.
    
        objFuncCellValues: the discrete value of objective on each cell source (index). 
        This will be used for computing df/dw in the adjoint.
    
        objFuncValue: the sum of objective, reduced across all processors and scaled by "scale"
    */

    // initialize cell values to zero
    forAll(objFuncCellValues, idxI)
    {
        objFuncCellValues[idxI] = 0.0;
    }
    // initialize objFunValue*/
    objFuncValue = 0.0; 
    const objectRegistry& db = mesh_.thisDb();

    if (data_ == "experimental")
    {
    if (targetPtr_.get()!=nullptr)
        {
            const fvMesh& target_ = targetPtr_();
            const objectRegistry& db_new = target_.thisDb();
            Info << "Checking for ubData in target's objectRegistry" << endl;

            if (db_new.foundObject<volVectorField>("ubData"))
            {
                const volVectorField& ubData = ubDataPtr_();//db_new.lookupObject<volVectorField>("ubData");
                Info << "ubData found with size: " << ubData.size() << endl;
                // allocate all box average lists
                List<scalar> volAvg(exp_final.size(), 0.0);
                List<scalar> uVolAvg(exp_final.size(), 0.0);
                List<scalar> vVolAvg(exp_final.size(), 0.0);
                List<scalar> wVolAvg(exp_final.size(), 0.0);
                List<scalar> uBoxAvg(exp_final.size(), 0.0);
                List<scalar> vBoxAvg(exp_final.size(), 0.0);
                List<scalar> wBoxAvg(exp_final.size(), 0.0);

                const volVectorField& U = db.lookupObject<volVectorField>("U");
                const scalarField volumes = mesh_.V();

                // do the box averaging 
                const scalarField ux = U.component(0);
                const scalarField uy = U.component(1);
                const scalarField uz = U.component(2);

                forAll(exp_final,idx)
                {
                    List<label> temp = cfd_final[idx];  
                    scalar avgU = 0.0;
                    scalar avgV = 0.0;
                    scalar avgW = 0.0;
                    scalar volavg = 0.0;
                    forAll(temp, idx1)
                    {
                        label tmp = temp[idx1];
                        avgU = avgU + volumes[tmp]*ux[tmp];
                        avgV = avgV + volumes[tmp]*uy[tmp];
                        avgW = avgW + volumes[tmp]*uz[tmp];
                        volavg = volavg + volumes[tmp];
                    }
                    volAvg[idx] = volavg;
                    uVolAvg[idx] = avgU;
                    vVolAvg[idx] = avgV;
                    wVolAvg[idx] = avgW;  
                    if (volAvg[idx] == 0.0)
                        {
                            uBoxAvg[idx] = 0.0;
                            vBoxAvg[idx] = 0.0;
                            wBoxAvg[idx] = 0.0;
                        }
                    else
                        {
                            uBoxAvg[idx] = uVolAvg[idx]/volAvg[idx];
                            vBoxAvg[idx] = vVolAvg[idx]/volAvg[idx];
                            wBoxAvg[idx] = wVolAvg[idx]/volAvg[idx];
                        }
                }
                forAll(exp_final, idx)
                {
                    const label& cellI = exp_final[idx]; 
                    List<label>& temp = cfd_final[idx];
                    const scalar& ub = ubData[cellI].component(0);
                    const scalar& vb = ubData[cellI].component(1);
                    if (temp.size() > 0 && ub!=0 && vb!=0)
                    {
                        forAll(temp, idx1)
                        {
                            label tempTemp = temp[idx1];
                            objFuncCellValues[tempTemp] = 0.5*(sqr(scale_ * uBoxAvg[idx] - ubData[cellI].component(0)) + 
                                                            sqr(scale_ * vBoxAvg[idx] - ubData[cellI].component(1)));
                        }
                    }
                }
                forAll(objFuncCellValues, idx)
                {
                    objFuncValue = objFuncValue + objFuncCellValues[idx];
                }

                reduce(objFuncValue, sumOp<scalar>());
            }
            else
            {
                FatalErrorInFunction
                    << "request for volVectorField ubData from objectRegistry target failed"
                    << nl << "Available objects of type volVectorField are "
                    << db_new.names("volVectorField") << endl;
            }
        }
        else
        {
            // Handle the case where targetPtr_ is not initialized
            Info << "use3D is set to no; target mesh is not initialized." << endl;
        }
    }
    
    else if (data_ == "gradient")
    {
        stateName_ = "fvSource";
        const volVectorField fvSource_ = db.lookupObject<volVectorField>(stateName_);

        volScalarField regularise = sqr(mag(fvc::grad(fvSource_)));
        scalar sum = 0;
        forAll(regularise, idxI)
        {
            if (regularise[idxI]!= 0)
                sum+=1;
            else
                continue;
        }

        Info << "The regularisation field has "<< sum << " non-zero entries in it" << endl;
        
        forAll(objFuncCellSources, idxI)
        {
            const label& cellI = objFuncCellSources[idxI];
            objFuncCellValues[idxI] = 0.5 * scale_ * scale_ * (regularise[cellI]);
            objFuncValue += objFuncCellValues[idxI];
        }
        // need to reduce the sum of all objectives across all processors
        reduce(objFuncValue, sumOp<scalar>());

        if (weightedSum_ == true)
        {
            objFuncValue = weight_ * objFuncValue;
        }
    }

    else if (data_ == "source")
    {
        stateName_ = "fvSource";
        const volVectorField fvSource_ = db.lookupObject<volVectorField>(stateName_);

        volScalarField regularise_1 = sqr(mag(fvSource_));
        scalar sum_1 = 0;
        forAll(regularise_1, idxI)
        {
            if (regularise_1[idxI]!= 0)
                sum_1+=1;
            else
                continue;
        }

        Info << "The regularisation field has "<< sum_1 << " non-zero entries in it" << endl;
        
        forAll(objFuncCellSources, idxI)
        {
            const label& cellI = objFuncCellSources[idxI];
            objFuncCellValues[idxI] = 0.5 * scale_ * scale_ * (regularise_1[cellI]);
            objFuncValue += objFuncCellValues[idxI];
        }
        // need to reduce the sum of all objectives across all processors
        reduce(objFuncValue, sumOp<scalar>());

        if (weightedSum_ == true)
        {
            objFuncValue = weight_ * objFuncValue;
        }
    }
 
    else if (data_ == "UData")
    {

    stateName_ = "U";
    
    forAll(objFuncCellValues, idx)
    {
        objFuncValue = objFuncValue + objFuncCellValues[idx];
    }

    reduce(objFuncValue, sumOp<scalar>());


    if (weightedSum_ == true)
        {
            objFuncValue = weight_ * objFuncValue;
        }
    
   
    }
    else
    {
        FatalErrorIn("") << "dataType: " << data_
                            << " not supported for field inversion! "
                            << "Available options are: UData, pData, surfacePressureData, surfaceFrictionData, aeroCoeffData, and surfaceFrictionDataPeriodicHill."
                            << abort(FatalError);
    }

    return;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //