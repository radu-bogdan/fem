/*

My own FESpace for linear and quadratic triangular elements.

A fe-space provides the connection between the local reference
element, and the global mesh.

*/

#include <comp.hpp> // provides FESpace, ...
#include <python_comp.hpp>

#include "hdivlumpingfe.hpp"
#include "hdivlumpingfespace.hpp"

namespace ngcomp
{
  /*
     the MeshAccess object provides information about the finite element mesh,
     see: https://github.com/NGSolve/ngsolve/blob/master/comp/meshaccess.hpp

     base class FESpace is here:
     https://github.com/NGSolve/ngsolve/blob/master/comp/fespace.hpp
  */

  HDivLumpingFESpace ::HDivLumpingFESpace(shared_ptr<MeshAccess> ama, const Flags &flags)
      : FESpace(ama, flags), verbose(false)
  {
    cout << "Constructor of HDivLumpingFESpace" << endl;
    cout << "Flags = " << flags << endl;

    verbose = flags.GetDefineFlag("verbose");

    type = "HDivLumpingFESpace";

    evaluator[VOL] = make_shared<T_DifferentialOperator<DiffOpIdHDiv<2>>>();
    evaluator[BND] = make_shared<T_DifferentialOperator<DiffOpIdVecHDivBoundary<2>>>();
    flux_evaluator[VOL] = make_shared<T_DifferentialOperator<DiffOpDivHDiv<2>>>();
  }

  DocInfo HDivLumpingFESpace ::GetDocu()
  {
    auto docu = FESpace::GetDocu();
    docu.short_docu = "My own FESpace.";
    docu.long_docu =
        R"raw_string(My own FESpace provides first and second order triangular elements.
)raw_string";
    docu.Arg("verbose") = "bool = False\n"
                          "  Display debug messages.";
    return docu;
  }

  void HDivLumpingFESpace ::Update()
  {
    // some global update:
    cout << "Update HDivLumpingFESpace, #edges = " << ma->GetNEdges()
         << ", #elements = " << ma->GetNE(VOL) << endl;

    // number of vertices
    nedges = ma->GetNEdges();
    nel = ma->GetNE();

    // for (auto && e : ma->Edges())
    //   cout <<  ma->GetElement(e) << endl;

    // number of dofs:
    SetNDof(2 * nedges + 2 * nel);
  }

  /*
    returns dof-numbers of element with ElementId ei
    element may be a volume element, or boundary element
  */
  void HDivLumpingFESpace ::GetDofNrs(ElementId ei, Array<DofId> &dnums) const
  {
    dnums.SetSize(0);

    auto vnum = ma->GetElement(ei).Vertices();
    auto ednum = ma->GetElement(ei).Edges();
    const EDGE *edges = ElementTopology::GetEdges(ET_TRIG);

    int permut[3] = {0, 1, 2};

    // if (vnum[0] > vnum[1])
    // if (vnum[1] > vnum[2])
    //   std::swap(permut[1], permut[2]);
    if (ei.IsVolume() && verbose)
    {
      cout << ei.Nr() << ": [";
      for (auto v : vnum)
        cout << v << " ";
      cout << "]" << endl;

      for (int i = 0; i < 3; i++)
        cout << "edge" << i << "(" << ednum[i] << ") : {" << vnum[edges[i][0]] << "," << vnum[edges[i][1]] << "}" << endl;
    }

    int i = 0;
    for (auto e : ednum)
    {
        dnums.Append(e);
        dnums.Append(nedges + e);
    }

    if (ei.IsVolume())
    {
      dnums.Append(2 * nedges + 2*ei.Nr());
      dnums.Append(2 * nedges + 2*ei.Nr()+1);
    }
  }

  /*
    Allocate finite element class, using custom allocator alloc
  */
  FiniteElement &HDivLumpingFESpace ::GetFE(ElementId ei, Allocator &alloc) const
  {
    switch (ma->GetElement(ei).GetType())
    {
    case ET_TRIG:
    {
      auto fe = new (alloc) HDivLumpingTrig;
      fe->SetVertexNumbers(ma->GetElVertices(ei));
      return *fe;
    }
    default:
      throw Exception("HDivLumpingFESpace: Element of type " + ToString(ma->GetElement(ei).GetType()) +
                      " not available\n");
    }
  }

  void HDivLumpingFESpace ::GetEdgeDofNrs(int ednr, Array<int> &dnums) const
  {
    dnums.SetSize0();

    dnums += ednr;
    dnums += ednr + nedges;
  }

}

void ExportHDivLumpingFESpace(py::module m)
{
  using namespace ngcomp;

  cout << "called ExportHDivLumpingFESpace" << endl;

  ExportFESpace<HDivLumpingFESpace>(m, "HDivLumpingFESpace", true)
      .def("GetNEdges", &HDivLumpingFESpace::GetNEdges,
              "return number of edges")
      .def("GetNE", &HDivLumpingFESpace::GetNE,
           "return number of elements");
}
