const fvMesh& target = targetPtr_();
const fvMesh& mesh = meshPtr_();

const vectorField centers = mesh.C();

scalarList cx(centers.size());
scalarList cy(centers.size());
scalarList cz(centers.size());

cx = centers.component(0);
cy = centers.component(1);
cz = centers.component(2);

// intersected cell labels
List<label> exp_cells;
List<label> cfd_cells;

meshSearch ms(target);

forAll(cx, idxI)
{
point pointCoord = {cx[idxI], cy[idxI], cz[idxI]};

// take a CFD point and tell which experimental cell does this lie inside
label cellI = ms.findCell(pointCoord, 0, true);
if (cellI >= 0)
{
 cfd_cells.append(idxI);
 exp_cells.append(cellI);            
}
}

// Use a hash table to group CFD cells by experimental cells
Foam::HashTable<List<label>, Foam::label, Foam::Hash<label>> cellGroups(exp_cells.size());

for (label i = 0; i < exp_cells.size(); i++)
{
if (!cellGroups.found(exp_cells[i]))
{
 cellGroups.insert(exp_cells[i], List<label>());
}
cellGroups[exp_cells[i]].append(cfd_cells[i]);
}

DAObjFunc::exp_final.setSize(exp_cells.size());
DAObjFunc::cfd_final.setSize(exp_cells.size());


for (Foam::HashTable<List<label>, Foam::label, Foam::Hash<label>>::iterator it = cellGroups.begin(); it != cellGroups.end(); ++it)
{
DAObjFunc::exp_final.append(it.key());
DAObjFunc::cfd_final.append(it());
}
