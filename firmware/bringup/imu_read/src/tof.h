#ifndef TOF_H
#define TOF_H

#include <stdint.h>
#include <stdbool.h>

struct tof_sample {
	float height;      // alpha-beta filtered, metres
	float vz;          // alpha-beta filtered, metres/second
	float height_raw;  // unfiltered measurement, metres
	float vz_raw;      // raw finite difference, m/s
	int64_t timestamp; // k_uptime_get() ms when this sample was taken
};

// A sample older than this is not trustworthy. ~6 missed conversions at the
// sensor's ~30Hz rate.
#define TOF_SAMPLE_MAX_AGE_MS 200

// Copy the most recent sample out.
void tof_get_latest(struct tof_sample *out);

// A dead ToF thread might leave the sample stale, but it will never block the control loop.
// The sample is always valid to read, but it may be stale.
// Return true if the sample is fresh enough to trust, false if it is stale.
bool tof_sample_fresh(const struct tof_sample *s);

#endif // TOF_H
