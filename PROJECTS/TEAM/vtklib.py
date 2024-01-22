import vtk
import numpy as np

def createVTK(MESH):
    
    points = vtk.vtkPoints()
    grid = vtk.vtkUnstructuredGrid()

    for i in range(MESH.np): points.InsertPoint(i, (MESH.p[i,0], MESH.p[i,1], MESH.p[i,2]))
        
    def create_cell(i):
        tetra = vtk.vtkTetra()
        ids = tetra.GetPointIds()
        ids.SetId(0, MESH.t[i,0])
        ids.SetId(1, MESH.t[i,1])
        ids.SetId(2, MESH.t[i,2])
        ids.SetId(3, MESH.t[i,3])
        return tetra

    elems = [create_cell(i) for i in range(MESH.nt)]
    grid.Allocate(MESH.nt, 1)
    grid.SetPoints(points)

    for elem in elems: grid.InsertNextCell(elem.GetCellType(), elem.GetPointIds())

    scalars = MESH.t[:,-1]
    data = vtk.vtkDoubleArray()
    data.SetNumberOfValues(MESH.nt)
    for i,p in enumerate(scalars): data.SetValue(i,p)
    grid.GetCellData().SetScalars(data)
    
    return grid


def add_H1_Scalar(grid,x,name):
    np = grid.GetPoints().GetNumberOfPoints()
    vecJ = vtk.vtkFloatArray()
    vecJ.SetNumberOfValues(np)
    vecJ.SetNumberOfComponents(1)
    for i in range(np):
        vecJ.SetValue(i,x[i])
    vecJ.SetName(name)
    grid.GetPointData().AddArray(vecJ)


def add_L2_Vector(grid,x,y,z,name):
    nt = grid.GetCells().GetNumberOfCells()
    vecJ = vtk.vtkFloatArray()
    vecJ.SetNumberOfComponents(3)
    for i in range(nt):
        vecJ.InsertNextTuple([x[i],y[i],z[i]])
    vecJ.SetName(name)
    grid.GetCellData().AddArray(vecJ)


def add_L2_Scalar(grid,x,name):
    nt = grid.GetCells().GetNumberOfCells()
    vecJ = vtk.vtkFloatArray()
    vecJ.SetNumberOfComponents(1)
    for i in range(nt):
        vecJ.InsertNextTuple([x[i]])
    vecJ.SetName(name)
    grid.GetCellData().AddArray(vecJ)
    
def writeVTK(grid,name):
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(name)
    writer.SetInputData(grid)
    writer.Write()
