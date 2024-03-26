
#include "hdivlumpingfe.hpp"

namespace ngfem
{

  void HDivLumpingTrig::CalcShape(const IntegrationPoint &ip,
                                                   SliceMatrix<> shape) const
  {
     /*
      Vertex coordinates have been defined to be (1,0), (0,1), (0,0)
      see file 
      https://github.com/NGSolve/ngsolve/blob/master/fem/elementtopology.cpp
      ElementTopology::GetVertices(ET_TRIG)
     */

    AutoDiff<2> x(ip(0), 0); // value of x, gradient is 0-th unit vector (1,0)
    AutoDiff<2> y(ip(1), 1); // value of y, gradient is 1-th unit vector (0,1)
    AutoDiff<2> lam[3] = {x, y, 1 - x - y};
    const EDGE * edges = ElementTopology::GetEdges (ET_TRIG);

    AutoDiff<2> bub1[2] = {lam[1] * (  lam[0] * lam[2].DValue(1) - lam[2] * lam[0].DValue(1) ),
                           lam[1] * ( -lam[0] * lam[2].DValue(0) + lam[2] * lam[0].DValue(0) )};
    AutoDiff<2> bub2[2] = {lam[2] * (  lam[0] * lam[1].DValue(1) - lam[1] * lam[0].DValue(1) ),
                           lam[2] * ( -lam[0] * lam[1].DValue(0) + lam[1] * lam[0].DValue(0) )};

    Vec<2> psi[3][2];
    psi[2][0] = { lam[0].Value() * lam[1].DValue(1) + bub1[0].Value() - 2 * bub2[0].Value(),
                 -lam[0].Value() * lam[1].DValue(0) + bub1[1].Value() - 2 * bub2[1].Value()};
    psi[2][1] = { ( lam[1].Value() * lam[0].DValue(1) + bub1[0].Value() + bub2[0].Value()),
                  (-lam[1].Value() * lam[0].DValue(0) + bub1[1].Value() + bub2[1].Value())};
    psi[1][0] = { lam[1].Value() * lam[2].DValue(1) - 2 * bub1[0].Value() + bub2[0].Value(),
                 -lam[1].Value() * lam[2].DValue(0) - 2 * bub1[1].Value() + bub2[1].Value()};
    psi[1][1] = { (lam[2].Value() * lam[1].DValue(1) + bub1[0].Value() - 2 * bub2[0].Value()),
                  (-lam[2].Value() * lam[1].DValue(0) + bub1[1].Value() - 2 * bub2[1].Value())};
    psi[0][1] = { ( lam[0].Value() * lam[2].DValue(1) - 2 * bub1[0].Value() + bub2[0].Value()),
                  (-lam[0].Value() * lam[2].DValue(0) - 2 * bub1[1].Value() + bub2[1].Value())};
    psi[0][0] = { lam[2].Value() * lam[0].DValue(1) + bub1[0].Value() + bub2[0].Value(),
                 -lam[2].Value() * lam[0].DValue(0) + bub1[1].Value() + bub2[1].Value()};

    int o = 0;
    for (int i = 0; i < 3; i++)
    {
      if (vnums[edges[i][0]] > vnums[edges[i][1]])
      {
        shape.Row(o++) = psi[i][1];
        shape.Row(o++) = psi[i][0];
      }
      else
      {
        shape.Row(o++) = psi[i][0];
        shape.Row(o++) = psi[i][1];
      }
    }

    // bubles
    shape(6,0) = bub1[0].Value();
    shape(6,1) = bub1[1].Value();
    // bubbles
    shape(7,0) = bub2[0].Value();
    shape(7,1) = bub2[1].Value();
  }

}