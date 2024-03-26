#ifndef FILE_HDIVLUMPINGFE_HPP
#define FILE_HDIVLUMPINGFE_HPP


#include<fem.hpp>
/*

My own simple first and second order triangular finite elements

*/

namespace ngfem
{

    class NGS_DLL_HEADER HDivLumpingTrig : public HDivFiniteElement<2>, public VertexOrientedFE<ET_TRIG>
    {
    protected:
    public:
        HDivLumpingTrig() : HDivFiniteElement(8, 2) {}

        using VertexOrientedFE<ET_TRIG>::SetVertexNumbers;

        ELEMENT_TYPE ElementType() const override { return ET_TRIG; }

        virtual void CalcShape(const IntegrationPoint &ip,
                               SliceMatrix<> shape) const override;
    };
}

#endif // FILE_HDIVLUMPINGFE_HPP
