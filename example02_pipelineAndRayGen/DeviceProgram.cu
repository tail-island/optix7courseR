#include <optix_device.h>

#include "Params.h"

/*! launch parameters in constant memory, filled in by optix upon
      optixLaunch (this gets filled in from the buffer we pass to
      optixLaunch) */

__constant__ osc::LaunchParams OptixLaunchParams;

namespace osc {

//------------------------------------------------------------------------------
// ray gen program - the actual rendering happens in here
//------------------------------------------------------------------------------

extern "C" __global__ void __raygen__renderFrame() {
  if (optixGetLaunchIndex().x == 0 && optixGetLaunchIndex().y == 0) {
    // we could of course also have used optixGetLaunchDims to query
    // the launch size, but accessing the optixLaunchParams here
    // makes sure they're not getting optimized away (because
    // otherwise they'd not get used)
    printf("############################################\n");
    printf("Hello world from OptiX 7 raygen program!\n(within a %ix%i-sized launch)\n", OptixLaunchParams.Width, OptixLaunchParams.Height);
    printf("############################################\n");
  }

  // ------------------------------------------------------------------
  // for this example, produce a simple test pattern:
  // ------------------------------------------------------------------

  // compute a test pattern based on pixel ID
  const auto& X = optixGetLaunchIndex().x;
  const auto& Y = optixGetLaunchIndex().y;

  const auto R = X % 256;
  const auto G = Y % 256;
  const auto B = (X + Y) % 256;

  // convert to 32-bit rgba value (we explicitly set alpha to 0xff
  // to make stb_image_write happy ...

  // and write to frame buffer ...
  OptixLaunchParams.ImageBuffer[X + Y * OptixLaunchParams.Width] = 0xff000000 | R << 0 | G << 8 | B << 16;
}

//------------------------------------------------------------------------------
// closest hit and anyhit programs for radiance-type rays.
//
// Note eventually we will have to create one pair of those for each
// ray type and each geometry type we want to render; but this
// simple example doesn't use any actual geometries yet, so we only
// create a single, dummy, set of them (we do have to have at least
// one group of them to set up the SBT)
//------------------------------------------------------------------------------

extern "C" __global__ void __closesthit__radiance() {
  ; // とりあえず、なにもしません。
}

extern "C" __global__ void __anyhit__radiance() {
  ; // とりあえず、なにもしません。
}

//------------------------------------------------------------------------------
// miss program that gets called for any ray that did not have a
// valid intersection
//
// as with the anyhit/closest hit programs, in this example we only
// need to have _some_ dummy function to set up a valid SBT
// ------------------------------------------------------------------------------

extern "C" __global__ void __miss__radiance() {
  ; // とりあえず、なにもしません。
}

} // namespace osc
