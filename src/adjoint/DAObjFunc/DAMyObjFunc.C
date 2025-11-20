/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v3

\*---------------------------------------------------------------------------*/

#include "DAMyObjFunc.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

using namespace std;
namespace Foam

{

defineTypeNameAndDebug(DAMyObjFunc, 0);
addToRunTimeSelectionTable(DAObjFunc, DAMyObjFunc, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DAMyObjFunc::DAMyObjFunc(
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


    
    // the spacing of the cells in x and y (ideally dx = dy)
    scalar dx = daOption_.getOption<scalar>("dx");
    scalar dy = daOption_.getOption<scalar>("dy");
    
    // starting coordinate of x and y of experimental grid
    scalar start_x = daOption_.getOption<scalar>("start_x");
    scalar start_y = daOption_.getOption<scalar>("start_y");
    
    // number of cells in X and Y
    label Ny = daOption_.getOption<label>("exp_rows");
    label Nx = daOption_.getOption<label>("exp_columns");

    // for 3D cases
    scalar zmin = daOption_.getOption<scalar>("zmin");
    scalar zmax = daOption_.getOption<scalar>("zmax");


    const vectorField centers = mesh_.C();

    scalarList cx(centers.size());
    scalarList cy(centers.size());
    scalarList cz(centers.size());
    
    cx = centers.component(0);
    cy = centers.component(1);
    cz = centers.component(2);
     
    // create the experimental mesh (not exactly create using fvMesh class but just for computation)
    List<scalar> xpts;
    List<scalar> ypts;
  
    xpts.append(start_x);
    ypts.append(start_y);
    
    for (label i = 1; i < Nx + 1; i++)
        xpts.append(xpts[i - 1] + dx);
    
    for (label i = 1; i < Ny + 1; i++)
        ypts.append(ypts[i - 1] + dy);
         
   // store points of CFD that lie in experimental cell
   List<label> temp;
   label searches = 0;
   for (label j = 0;j < ypts.size() -1; j++)
        {
	    for (label i= 0 ; i < xpts.size() - 1 ; i++)
	        {
            forAll(cx, cellI)
            {
                searches+=1;
		        if (cx[cellI] <= xpts[i + 1] and cx[cellI] >= xpts[i] and cy[cellI] <= ypts[j + 1] and cy[cellI] >= ypts[j] and cz[cellI]>zmin and cz[cellI]<zmax)
				    temp.append(cellI);			
		    }
		    points.append(temp);
		    temp.clear();
	        }

        }
    Info << "The number of searches completed is " << searches << endl;
    
    

}

/// calculate the value of objective function
void DAMyObjFunc::calcObjFunc(
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

    
    // get user input for U and V velocities
    scalarList UAv = daOption_.getOption<scalarList>("UAv");
    scalarList VAv = daOption_.getOption<scalarList>("VAv");
    
    if (data_ == "gradient")
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
    

    const scalarField volumes = mesh_.V();
    const volVectorField state = db.lookupObject<volVectorField>(stateName_);

   
    scalarList ux(state.size());
    scalarList uy(state.size());
    scalarList uz(state.size());
  
    ux = state.component(0);
    uy = state.component(1);
    uz = state.component(2);
 

    // allocate all box average lists
    List<scalar> volAvg(points.size());
    List<scalar> uVolAvg(points.size());
    List<scalar> vVolAvg(points.size());
    List<scalar> wVolAvg(points.size());
    List<scalar> uBoxAvg(points.size());
    List<scalar> vBoxAvg(points.size());
    List<scalar> wBoxAvg(points.size());


    // and assign them with zeros 
    forAll(volAvg, idx)
    {
        volAvg[idx] = 0.0;
        uVolAvg[idx] = 0.0;
        vVolAvg[idx] = 0.0;
        wVolAvg[idx] = 0.0;
        uBoxAvg[idx] = 0.0;
        vBoxAvg[idx] = 0.0;
        wBoxAvg[idx] = 0.0;
    }
    
    // do the box averaging (but division with volume not done yet)
    forAll(points,idx)
    {
        List<label> temp = points[idx];
        
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
    }

    // do the division with volumes. if there are no points, set it to zero
    forAll(volAvg, idx)
    {
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

    // if there are no points, set obj func to zero, otherwise calculate it on experimental grid
    // and broadcast it to CFD grid simultaneously
    forAll(points, idx)
    {
        labelList temp = points[idx];
        //scalar reference(sqrt(sqr(UAv[idx]) + sqr(VAv[idx])));
        if (temp.size() > 0 && UAv[idx]!=0.0)
        {
            forAll(temp, idx1)
            {
                label tempTemp = temp[idx1];
                objFuncCellValues[tempTemp] = 0.5*(sqr(scale_ * uBoxAvg[idx] - UAv[idx]) + sqr(scale_ * vBoxAvg[idx] - VAv[idx]));
            }
        }
    }

    forAll(objFuncCellValues, idx)
    {
        objFuncValue = objFuncValue + objFuncCellValues[idx];
    }

    reduce(objFuncValue, sumOp<scalar>());


    if (weightedSum_ == true)
        {
            objFuncValue = weight_ * objFuncValue;
        }
    
    scalar Ubulk = daOption_.getOption<scalar>("Ubulk");
    /*scalar norm = 0.0;
    forAll(uBoxAvg, idx)
    {
        if (uBoxAvg[idx]!=0)
            norm = norm + (1/Ubulk)*(abs(uBoxAvg[idx] - UAv[idx]) + abs(vBoxAvg[idx] - VAv[idx]));
    }
    Info <<"The L1 norm during primal solver call is: "<< norm/points.size()<<endl;
    */
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