/*********************************************************************************************
Date: 04/05/2023
Author: Uttam Cadambi Padmanaban

This code does the box averaging and spits out the indices, intersection volumes and volume 
weights. 

Indices: a list of lists that contain the indices of all the CFD cells that intersect with 
every experimental cell. 

Intersection volumes: a list of lists that contain the intersection volumes of each 
experimental cell with the cells that hold the Indices of CFD volumes as mentioned in the 
previous description.

Volume weights: what is the percentage of a CFD volume that intersects with a given 
experimental cell? This is done to broadcast the objective function calculated on the 
experimental grid back to the CFD grid.

*********************************************************************************************/

// original mesh pointer dereferenced
fvMesh& mesh = meshPtr_();

// structured mesh pointer dereferenced 
fvMesh& mesh1 = meshPtr1_();

// to hold intersection volumes
tmpVol.setSize(mesh1.nCells());

// to hold indices
ind.setSize(mesh1.nCells());

// to hold weights
weights.setSize(mesh1.nCells());


// done to prevent junk values 
forAll(mesh1.cells(), idx)
{
    tmpVol[idx] = 0.0;
}

// interpolator object of directMethod class
directMethod interpolator(mesh, mesh1);

// for every experimental cell, run through all CFD cells
forAll(mesh1.cells(), idx2)
{
    scalarList vol(mesh.nCells());
    scalar tempVol = 0.0;
    forAll(mesh.cells(), idx)
    {
        // check if there is intersection
        vol[idx] = interpolator.interVol(idx, idx2);
        if (vol[idx] > 0)
        {
            tempVol = tempVol + vol[idx];
            ind[idx2].append(idx);
            weights[idx2].append(vol[idx]/mesh.V()[idx]);
            tmpVol[idx2].append(vol[idx]);
        }
    }
    vol.clear();
}

// initialize the static variables from DASolver class.
// this is done because we want to be able to access the variables 
// regardless of the instantiation. Before that we need to set the 
// sizes so that they are contiguous blocks of memeory

DAObjFunc::indices.setSize(ind.size());
DAObjFunc::intersectVols.setSize(tmpVol.size());
DAObjFunc::volWeights.setSize(weights.size());

forAll(mesh1.cells(),idx)
{
    DAObjFunc::indices[idx].append(ind[idx]);
    DAObjFunc::intersectVols[idx].append(tmpVol[idx]);
    DAObjFunc::volWeights[idx].append(weights[idx]);
}

