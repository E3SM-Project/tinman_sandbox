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
      v2[j][l] - dsdy * rrearth;
    }
  }

  for (int j=0; j<np; ++j)
  {
    for (int i=0; i<np; ++i)
    {
      AT_3D (ds, np, 2, i, j, 0) = AT_4D(Dinv,i,j,0,0,np,2,2)*v1[i][j]
                                 + AT_4D(Dinv,i,j,1,0,np,2,2)*v1[i][j];

      AT_3D (ds, np, 2, i, j, 1) = AT_4D(Dinv,i,j,0,1,np,2,2)*v1[i][j]
                                 + AT_4D(Dinv,i,j,1,1,np,2,2)*v1[i][j];
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
  for (int ipt=0; ipt<np; ++ipt)
  {
    for (int jpt=0; jpt<np; ++jpt)
    {
      gv[ipt][jpt][0] = AT_2D(metdet,ipt,jpt,np)* ( AT_4D(Dinv,ipt,jpt,0,0,np,2,2)*AT_3D(v,ipt,jpt,0,np,2)
                                                   +AT_4D(Dinv,ipt,jpt,0,1,np,2,2)*AT_3D(v,ipt,jpt,1,np,2) );
      gv[ipt][jpt][1] = AT_2D(metdet,ipt,jpt,np)* ( AT_4D(Dinv,ipt,jpt,1,0,np,2,2)*AT_3D(v,ipt,jpt,0,np,2)
                                                   +AT_4D(Dinv,ipt,jpt,1,1,np,2,2)*AT_3D(v,ipt,jpt,1,np,2) );
    }
  }

  real dudx, dvdy;
  for (int ipt=0; ipt<np; ++ipt)
  {
    for (int jpt=0; jpt<np; ++jpt)
    {
      dudx = dvdy = 0.;
      for (int kpt=0; kpt<np; ++kpt)
      {
        dudx += Dvv[kpt][ipt] * gv[kpt][jpt][0];
        dvdy += Dvv[kpt][jpt] * gv[ipt][kpt][1];
      }

      AT_2D(div,ipt,jpt,np) = (dudx + dvdy) * AT_2D(rmetdet,ipt,jpt,np) * rrearth;
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
  for (int ipt=0; ipt<np; ++ipt)
  {
    for (int jpt=0; jpt<np; ++jpt)
    {
      vcov[ipt][jpt][0] = AT_4D(D,ipt,jpt,0,0,np,2,2)*AT_3D(v,ipt,jpt,0,np,2)
                        + AT_4D(D,ipt,jpt,1,0,np,2,2)*AT_3D(v,ipt,jpt,1,np,2);
      vcov[ipt][jpt][1] = AT_4D(D,ipt,jpt,0,1,np,2,2)*AT_3D(v,ipt,jpt,0,np,2)
                        + AT_4D(D,ipt,jpt,1,1,np,2,2)*AT_3D(v,ipt,jpt,1,np,2);
    }
  }

  real dudy, dvdx;
  for (int ipt=0; ipt<np; ++ipt)
  {
    for (int jpt=0; jpt<np; ++jpt)
    {
      dudy = dvdx = 0.;
      for (int kpt=0; kpt<np; ++kpt)
      {
        dudy += Dvv[kpt][jpt] * vcov[ipt][kpt][1];
        dvdx += Dvv[kpt][ipt] * vcov[kpt][jpt][0];
      }

      AT_2D(vort,ipt,jpt,np) = (dvdx - dudy) * AT_2D(rmetdet,ipt,jpt,np) * rrearth;
    }
  }
}

} // Namespace Homme
