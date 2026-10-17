#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#ifdef HAVE_OPENCL
#include "common/ekvido/cl/ekv_cl_dt.h"
#include "common/darktable.h"
#include "common/dlopencl.h"
#include "common/file_location.h"
#include "common/opencl.h"

#include <glib.h>
#include <glib/gstdio.h>
#include <limits.h>
#include <stdlib.h>

#define EKV_DT_MAX_DEV 16

G_LOCK_DEFINE_STATIC(ekv_dt);
static ekv_cl_t *_dev[EKV_DT_MAX_DEV];
static gboolean _failed[EKV_DT_MAX_DEV];
static int _users = 0;
static ekv_clfn_t _fn;
static gboolean _fn_set = FALSE;

void ekv_cl_dt_retain(void)
{
  G_LOCK(ekv_dt);
  _users++;
  G_UNLOCK(ekv_dt);
}

void ekv_cl_dt_release(void)
{
  G_LOCK(ekv_dt);
  if(--_users == 0)
    for(int i = 0; i < EKV_DT_MAX_DEV; i++)
    {
      ekv_cl_free(_dev[i]);
      _dev[i] = NULL;
      _failed[i] = FALSE;
    }
  G_UNLOCK(ekv_dt);
}

ekv_cl_t *ekv_cl_dt_get(const int devid)
{
#ifdef __APPLE__
  return NULL; /* spec D1: the Mac runs the CPU path (Apple's OpenCL is untested here) */
#endif
  if(devid < 0 || devid >= EKV_DT_MAX_DEV || !darktable.opencl || !darktable.opencl->dlocl) return NULL;
  G_LOCK(ekv_dt);
  if(!_dev[devid] && !_failed[devid])
  {
    const dt_dlopencl_symbols_t *s = darktable.opencl->dlocl->symbols;
    if(!_fn_set)
    {
#define EKV_FROM_DT(n) _fn.n = (__typeof__(_fn.n))s->dt_cl##n;
      EKV_CLFN_LIST(EKV_FROM_DT)
#undef EKV_FROM_DT
      ekv_cl_set_api(&_fn);
      _fn_set = TRUE;
    }
    char kdir[PATH_MAX] = { 0 }, cdir[PATH_MAX] = { 0 };
    dt_loc_get_kerneldir(kdir, sizeof(kdir));
    dt_loc_get_user_cache_dir(cdir, sizeof(cdir));
    gchar *path = g_build_filename(kdir, "ekvido.cl", NULL);
    gchar *plans = g_build_filename(cdir, "ekvido-vkfft", NULL);
    g_mkdir_with_parents(plans, 0700);
    char *source = ekv_cl_read_source(path);
    const dt_opencl_device_t *d = &darktable.opencl->dev[devid];
    _dev[devid] = source ? ekv_cl_new(d->context, d->devid, d->cmd_queue, source, plans) : NULL;
    _failed[devid] = _dev[devid] == NULL;
    if(_failed[devid])
      dt_print(DT_DEBUG_OPENCL, "[ekvido] GPU path unavailable on device %d (%s); using the CPU", devid, path);
    free(source);
    g_free(path);
    g_free(plans);
  }
  ekv_cl_t *E = _dev[devid];
  G_UNLOCK(ekv_dt);
  return E;
}
#endif /* HAVE_OPENCL */
