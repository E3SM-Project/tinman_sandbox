#include "sphere_operators.hpp"
#include "dimensions.hpp"
#include "test_macros.hpp"

namespace Homme
{

void gradient_sphere (const real* const s, const TestData& data,
                      int ielem, real* const ds)
{
  typedef real Dvv_type[np][np];

  const Dvv_type& Dvv = data.deriv.Dvv;
  real rrearth = data.constants.rrearth;
  real* Dinv = SLICE_5D (data.arrays.elem_Dinv,ielem,np,np,2,2);

  real dsdx, dsdy;
  real v1[np][np];
  real v2[np][np];
  for (int j=0; j<np; ++j)
  {
    for (int l=0; l<np; ++l)
    {
      dsdx = dsdy = 0;
      for (int i=0; i<np; ++i)
      {
        dsdx += Dvv[i][l]*AT_2D(s,i,j,np);
        dsdy += Dvv[i][l]*AT_2D(s,j,i,np);
      }

      v1[l][j] = dsdx * rrearth;
      v2[j][l] = dsdy * rrearth;
    }
  }

  for (int j=0; j<np; ++j)
  {
    for (int i=0; i<np; ++i)
    {
      AT_3D (ds, i, j, 0, np, 2) = AT_4D(Dinv,i,j,0,0,np,2,2)*v1[i][j]
                                 + AT_4D(Dinv,i,j,1,0,np,2,2)*v2[i][j];

      AT_3D (ds, i, j, 1, np, 2) = AT_4D(Dinv,i,j,0,1,np,2,2)*v1[i][j]
                                 + AT_4D(Dinv,i,j,1,1,np,2,2)*v2[i][j];
    }
  }
}

void divergence_sphere (const real* const v, const TestData& data,
                        int ielem, real* const div)
{
  typedef real Dvv_type[np][np];

  const Dvv_type& Dvv = data.deriv.Dvv;
  real rrearth = data.constants.rrearth;

  real* Dinv    = SLICE_5D (data.arrays.elem_Dinv,ielem,np,np,2,2);
  real* metdet  = SLICE_3D (data.arrays.elem_metdet,ielem,np,np);
  real* rmetdet = SLICE_3D (data.arrays.elem_rmetdet,ielem,np,np);

  real gv[np][np][2];
  for (int igp=0; igp<np; ++igp)
  {
    for (int jgp=0; jgp<np; ++jgp)
    {
      gv[igp][jgp][0] = AT_2D(metdet,igp,jgp,np)* ( AT_4D(Dinv,igp,jgp,0,0,np,2,2)*AT_3D(v,igp,jgp,0,np,2)
                                                   +AT_4D(Dinv,igp,jgp,0,1,np,2,2)*AT_3D(v,igp,jgp,1,np,2) );
      gv[igp][jgp][1] = AT_2D(metdet,igp,jgp,np)* ( AT_4D(Dinv,igp,jgp,1,0,np,2,2)*AT_3D(v,igp,jgp,0,np,2)
                                                   +AT_4D(Dinv,igp,jgp,1,1,np,2,2)*AT_3D(v,igp,jgp,1,np,2) );
    }
  }

  real dudx, dvdy;
  for (int igp=0; igp<np; ++igp)
  {
    for (int jgp=0; jgp<np; ++jgp)
    {
      dudx = dvdy = 0.;
      for (int kgp=0; kgp<np; ++kgp)
      {
        dudx += Dvv[kgp][igp] * gv[kgp][jgp][0];
        dvdy += Dvv[kgp][jgp] * gv[igp][kgp][1];
      }

      AT_2D(div,igp,jgp,np) = (dudx + dvdy) * AT_2D(rmetdet,igp,jgp,np) * rrearth;
    }
  }
}

void vorticity_sphere (const real* const v, const TestData& data,
                       int ielem, real* const vort)
{
  typedef real Dvv_type[np][np];

  const Dvv_type& Dvv = data.deriv.Dvv;
  real rrearth = data.constants.rrearth;

  real* D       = SLICE_5D (data.arrays.elem_D,ielem,np,np,2,2);
  real* rmetdet = SLICE_3D (data.arrays.elem_rmetdet,ielem,np,np);

  real vcov[np][np][2];
  for (int igp=0; igp<np; ++igp)
  {
    for (int jgp=0; jgp<np; ++jgp)
    {
      vcov[igp][jgp][0] = AT_4D(D,igp,jgp,0,0,np,2,2)*AT_3D(v,igp,jgp,0,np,2)
                        + AT_4D(D,igp,jgp,1,0,np,2,2)*AT_3D(v,igp,jgp,1,np,2);
      vcov[igp][jgp][1] = AT_4D(D,igp,jgp,0,1,np,2,2)*AT_3D(v,igp,jgp,0,np,2)
                        + AT_4D(D,igp,jgp,1,1,np,2,2)*AT_3D(v,igp,jgp,1,np,2);
    }
  }

  real dudy, dvdx;
  for (int igp=0; igp<np; ++igp)
  {
    for (int jgp=0; jgp<np; ++jgp)
    {
      dudy = dvdx = 0.;
      for (int kgp=0; kgp<np; ++kgp)
      {
        dudy += Dvv[kgp][jgp] * vcov[igp][kgp][1];
        dvdx += Dvv[kgp][igp] * vcov[kgp][jgp][0];
      }

      AT_2D(vort,igp,jgp,np) = (dvdx - dudy) * AT_2D(rmetdet,igp,jgp,np) * rrearth;
    }
  }
}

} // Namespace Homme
