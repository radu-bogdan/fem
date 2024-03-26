#ifndef FILE_HDIVLUMPINGFESPACE_HPP
#define FILE_HDIVLUMPINGFESPACE_HPP

/*

  My own FESpace for linear and quadratic triangular elements An
  fe-space is the connection between the local reference element, and
  the global mesh.

*/

namespace ngcomp
{
    class HDivLumpingFESpace : public FESpace
    {
        bool verbose;
        bool swap;
        size_t nedges, nel;

    public:
        /*
          constructor.
          Arguments are the MeshAccess view of the mesh data structure,
          and the kwargs from the Python constructor converted to C++ Flags.
        */
        HDivLumpingFESpace(shared_ptr<MeshAccess> ama, const Flags &flags);

        // a name for our new fe-space
        string GetClassName() const override { return "HDivLumpingFESpace"; }

        // documentation
        static DocInfo GetDocu();

        // organzize the FESpace, called after every mesh update
        void Update() override;

        // dof-numbers for element-id ei
        void GetDofNrs(ElementId ei, Array<DofId> &dnums) const override;

        virtual void GetEdgeDofNrs (int ednr, Array<DofId> & dnums) const override;

        // generate FiniteElement for element-id ei
        FiniteElement &GetFE(ElementId ei, Allocator &alloc) const override;

        size_t GetNEdges() { return nedges; }

        size_t GetNE() { return nel; }

    };

}

void ExportHDivLumpingFESpace(py::module m);

#endif
