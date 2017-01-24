#include "sphere_operators.hpp"
#include "dimensions.hpp"
#include "test_macros.hpp"

namespace Homme
{

void gradient_sphere (const real* RESTRICT const s, const TestData& RESTRICT data,
                      int ielem, real* RESTRICT const ds)
{
  typedef real Dvv_type[np][np];

  const Dvv_type& RESTRICT Dvv = data.deriv.Dvv;
  real* Dinv = SLICE_5D (data.arrays.elem_Dinv,ielem,np,np,2,2);

  real dsdx, dsdy;
  real v1[np][np];
  real v2[np][np];
  SIMD
  for (int j=0; j<np; ++j)
  {
    for (int l=0; l<np; ++l)
    {
      dsdy = 0.0;
      for (int i=0; i<np; ++i)
      {
        dsdy += Dvv[i][l]*AT_2D(s,j,i,np);
      }
      v2[j][l] = dsdy * Constants::rrearth;
    }
  }

  SIMD
  for (int l=0; l<np; ++l)
  {
    for (int j=0; j<np; ++j)
    {
      dsdx = 0.0;
      for (int i=0; i<np; ++i)
      {
        dsdx += Dvv[i][l]*AT_2D(s,i,j,np);
      }
      v1[l][j] = dsdx * Constants::rrearth;
    }
  }

  SIMD
  for (int j=0; j<np; ++j)
  {
    for (int i=0; i<np; ++i)
    {
      for(int k = 0; k < 2; ++k)
      {
        AT_3D (ds, i, j, k, np, 2) = (AT_4D(Dinv,i,j,0,k,np,2,2)*v1[i][j]
                                      + AT_4D(Dinv,i,j,1,k,np,2,2)*v2[i][j]);
      }
    }
  }
}

void divergence_sphere (const real* RESTRICT const v, const TestData& RESTRICT data,
                        int ielem, real* RESTRICT const div)
{
  typedef real Dvv_type[np][np];

  /* Requires 8 cache lines */
  real* RESTRICT Dinv = SLICE_5D (data.arrays.elem_Dinv,ielem,np,np,2,2);

  /* Requires 4 cache lines */
  real* RESTRICT metdet = SLICE_3D (data.arrays.elem_metdet,ielem,np,np);

  /* Requires 4 cache lines */
  real gv[np][np][2];
  SIMD
  for (int igp=0; igp<np; ++igp)
  {
    for (int jgp=0; jgp<np; ++jgp)
    {
      for(int kgp = 0; kgp < 2; ++kgp)
      {
        gv[igp][jgp][kgp] = AT_2D(metdet,igp,jgp,np)*(AT_4D(Dinv,igp,jgp,kgp,0,np,2,2)
                                                      *AT_3D(v,igp,jgp,0,np,2)
                                                      +AT_4D(Dinv,igp,jgp,kgp,1,np,2,2)
                                                      *AT_3D(v,igp,jgp,1,np,2));
      }
    }
  }

  /* Requires 2 cache lines */
  const Dvv_type& RESTRICT Dvv = data.deriv.Dvv;

  /* Requires 2 cache lines */
  real* RESTRICT rmetdet = SLICE_3D (data.arrays.elem_rmetdet,ielem,np,np);
  real dd_sum[np][np];
  SIMD
  for (int igp=0; igp<np; ++igp)
  {
    for (int jgp=0; jgp<np; ++jgp)
    {
      dd_sum[igp][jgp] = 0;
    }
  }
  SIMD
  for (int kgp=0; kgp<np; ++kgp)
  {
    for (int igp=0; igp<np; ++igp)
    {
      for (int jgp=0; jgp<np; ++jgp)
      {
        dd_sum[igp][jgp] += Dvv[kgp][jgp] * gv[kgp][jgp][1] + Dvv[kgp][igp] * gv[igp][kgp][0];
      }
    }
  }
  SIMD
  for (int igp=0; igp<np; ++igp)
  {
    for (int jgp=0; jgp<np; ++jgp)
    {
      AT_2D(div,igp,jgp,np) = dd_sum[igp][jgp] * AT_2D(rmetdet,igp,jgp,np) * Constants::rrearth;
    }
  }
}

void vorticity_sphere (const real* RESTRICT const v, const TestData& RESTRICT data,
                       int ielem, real* RESTRICT const vort)
{
  /* In full, vorticity_sphere requires:
   *   12 cache lines for input data
   *   6 cache lines for local buffers (can be put into SIMD registers)
   *   7/8s of a cache line for local integers and pointers
   * for a total of 18+19/12 cache lines on average; or typically 20
   * (assuming uniform PDF of local variable position on the stack)
   * Note that 1 cache line = 1 AVX 512 register; of which there are 32
   */

  typedef real Dvv_type[np][np];

  /* Requires 2 cache lines */
  const Dvv_type& RESTRICT Dvv = data.deriv.Dvv;

  /* Requires 8 cache lines */
  real* RESTRICT D = SLICE_5D (data.arrays.elem_D,ielem,np,np,2,2);

  /* Requires 4 cache lines */
  real vcov[np][np][2];
  SIMD
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

  /* Requires 2 cache lines */
  real dd_diff[np][np];
  SIMD
  for (int igp=0; igp<np; ++igp)
  {
    for (int jgp=0; jgp<np; ++jgp)
    {
      dd_diff[igp][jgp] = 0;
    }
  }
  SIMD
  for (int kgp=0; kgp<np; ++kgp)
  {
    for (int igp=0; igp<np; ++igp)
    {
      for (int jgp=0; jgp<np; ++jgp)
      {
        dd_diff[igp][jgp] += Dvv[kgp][jgp] * vcov[igp][kgp][1] - Dvv[kgp][igp] * vcov[kgp][jgp][0];
      }
    }
  }

  /* Requires 2 cache lines */
  real* RESTRICT rmetdet = SLICE_3D (data.arrays.elem_rmetdet,ielem,np,np);
  SIMD
  for(int igp=0; igp<np; ++igp)
  {
    for(int jgp=0; jgp<np; ++jgp)
    {
      AT_2D(vort,igp,jgp,np) = dd_diff[igp][jgp] * AT_2D(rmetdet,igp,jgp,np) * Constants::rrearth;
    }
  }
}

} // Namespace Homme
