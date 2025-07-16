import torch

try:
    # raise ImportError("Debug Test")
    import torchCompactRadius
    from torchCompactRadius.compactHashing.datastructure import CompactHashMap
    from torchCompactRadius import radiusSearch, neighborSearchExisting
    from torchCompactRadius.util import PointCloud, DomainDescription, SparseCOO, SparseCSR

except ImportError as e:
    # raise e
    print("torchCompactRadius not found, using fallback implementations.")

    from fallback import CompactHashMap, radiusSearch, PointCloud, DomainDescription, SparseCOO, SparseCSR
    
from dataclasses import dataclass

@dataclass#(slots = True)
class SparseNeighborhood:
    row: torch.Tensor
    col: torch.Tensor
    
    numRows: int
    numCols: int
    
    points_a: 'PointCloud'
    points_b: 'PointCloud'
    
    domain: 'DomainDescription'

def filterNeighborhoodByKind(particleState, sparseNeighborhood : SparseNeighborhood, which : str = 'normal'):
    if which == 'all':
        return sparseNeighborhood
    i = sparseNeighborhood.row
    j = sparseNeighborhood.col
    
    if which == 'fluid':
        maskA = particleState.kinds[i] == 0
        maskB = particleState.kinds[j] == 0
    elif which == 'boundary':
        maskA = particleState.kinds[i] == 1
        maskB = particleState.kinds[j] == 1
    elif which == 'boundaryToFluid':
        maskA = particleState.kinds[i] == 0
        maskB = particleState.kinds[j] == 1
    elif which == 'fluidToBoundary':
        maskA = particleState.kinds[i] == 1
        maskB = particleState.kinds[j] == 0
    elif which == 'ghostToFluid':
        maskA = particleState.kinds[i] == 0
        maskB = particleState.kinds[j] == 2
    elif which == 'fluidToGhost':
        maskA = particleState.kinds[i] == 2
        maskB = particleState.kinds[j] == 0
    elif which == 'normal':
        maskA = torch.ones_like(i, dtype = torch.bool)
        maskB = particleState.kinds[j] != 2
    elif which == 'fluidWBoundary':
        maskA = particleState.kinds[i] == 0
        maskB = torch.logical_or(particleState.kinds[j] == 1, particleState.kinds[j] == 0)
    elif which == 'boundaryWFluid':
        maskA = particleState.kinds[i] == 1
        maskB = torch.logical_or(particleState.kinds[j] == 1, particleState.kinds[j] == 0)
    elif which == 'noghost':
        maskA = particleState.kinds[i] != 2
        maskB = particleState.kinds[j] != 2
    else:
        raise ValueError(f'which = {which} not recognized')
        
    mask = torch.logical_and(maskA, maskB)
    
    i_filtered = i[mask]
    j_filtered = j[mask]
    # newNumRows = torch.sum(maskA)
    # newNumCols = torch.sum(maskB)
    
    return SparseNeighborhood(
        row = i_filtered,
        col = j_filtered,
        numRows = sparseNeighborhood.numRows,
        numCols = sparseNeighborhood.numCols,
        
        points_a = sparseNeighborhood.points_a,
        points_b = sparseNeighborhood.points_b,
        
        domain = sparseNeighborhood.domain,
    )

def evalEdgeKinds(kinds : torch.Tensor, i : torch.Tensor, j: torch.Tensor):
    out = i.new_ones(i.shape, dtype = torch.int32) * 10

    ki = kinds[i]
    kj = kinds[j]

    mask_ftf = torch.logical_and(ki == 0, kj == 0)
    mask_btf = torch.logical_and(ki == 0, kj == 1)
    mask_ftb = torch.logical_and(ki == 1, kj == 0)
    mask_btb = torch.logical_and(ki == 1, kj == 1)
    mask_ftg = torch.logical_and(ki == 2, kj == 0)

    out = torch.where(mask_ftf, 0, out)
    out = torch.where(mask_btf, 1, out)
    out = torch.where(mask_ftb, 2, out)
    out = torch.where(mask_btb, 3, out)
    out = torch.where(mask_ftg, 4, out)

    outSorted = torch.argsort(out, dim = 0)

    return out, outSorted, torch.sum(mask_ftf), torch.sum(mask_btf), torch.sum(mask_ftb), torch.sum(mask_btb), torch.sum(mask_ftg)

def neighborSearch(state, domain, config):
    points = PointCloud(state.positions, state.supports)
    fullAdjacency = radiusSearch(
        points,
        points,
        domain = domain,
        mode = 'superSymmetric',
        returnStructure=False,
        algorithm='compact'
    )
    sparseNeighborhood = SparseNeighborhood(
        row = fullAdjacency.row,
        col = fullAdjacency.col,
        numRows = fullAdjacency.numRows,
        numCols = fullAdjacency.numCols,
        points_a = points,
        points_b = points,
        domain = domain
    )
    indices, indicesSorted, numFluidToFluid, numBoundaryToFluid, numFluidToBoundary, numBoundaryToBoundary, numFluidToGhost = evalEdgeKinds(state.kinds, sparseNeighborhood.row, sparseNeighborhood.col)

    numNormal = numFluidToFluid + numBoundaryToFluid + numFluidToBoundary + numBoundaryToBoundary #+ numFluidToGhost

    sortedNeighbors = SparseNeighborhood(sparseNeighborhood.row[indicesSorted], sparseNeighborhood.col[indicesSorted], numRows = sparseNeighborhood.numRows, numCols = sparseNeighborhood.numCols, 
                                        points_a = sparseNeighborhood.points_a, points_b = sparseNeighborhood.points_b, domain = sparseNeighborhood.domain)



    normalIndices = (0, numNormal)
    fluidIndices = (0, numFluidToFluid)
    boundaryToFluidIndices = (fluidIndices[-1], fluidIndices[-1] + numBoundaryToFluid)
    fluidToBoundaryIndices = (boundaryToFluidIndices[-1], boundaryToFluidIndices[-1] + numFluidToBoundary)
    boundaryToBoundaryIndices = (fluidToBoundaryIndices[-1], fluidToBoundaryIndices[-1] + numBoundaryToBoundary)
    fluidToGhostIndices = (boundaryToBoundaryIndices[-1], boundaryToBoundaryIndices[-1] + numFluidToGhost)

    return sortedNeighbors


def coo_to_csr(coo: SparseCOO, isSorted: bool = False) -> SparseCSR:
    if not isSorted:
        neigh_order = torch.argsort(coo.row)
        row = coo.row[neigh_order]
        col = coo.col[neigh_order]
    else:
        row = coo.row
        col = coo.col

    # print(f'Converting COO To CSR for matrix of shape {coo.numRows} x {coo.numCols}')
    # print(f'Number of Entries: {row.shape[0]}/{col.shape[0]}')
    jj, nit = torch.unique(row, return_counts=True)
    nj = torch.zeros(coo.numRows, dtype=coo.row.dtype, device=coo.row.device)
    nj[jj] = nit
    # print(f'Number of neighbors: {nj} ({nj.sum()} total, shape {nj.shape})')

    indptr = torch.zeros(coo.numRows + 1, dtype=torch.int64, device=coo.row.device)
    indptr[1:] = torch.cumsum(nj, 0)
    # print(f'Row pointers: {indptr} ({indptr.shape})')
    indptr = indptr.int()
    indices = col
    rowEntries = nj

    return SparseCSR(indices, indptr, rowEntries, coo.numRows, coo.numCols)