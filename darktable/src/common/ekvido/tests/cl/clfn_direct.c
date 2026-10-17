/* The standalone tests link libOpenCL: fill ekvido's table with the real entry points. */
#include "common/ekvido/cl/ekv_clfn.h"

void ekv_cl_use_linked_api(void)
{
  static ekv_clfn_t fn;
#define EKV_DIRECT(n) fn.n = &cl##n;
  EKV_CLFN_LIST(EKV_DIRECT)
#undef EKV_DIRECT
  ekv_cl_set_api(&fn);
}
