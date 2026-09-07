#include <zephyr/kernel.h>
#include <zephyr/device.h>
#include <zephyr/drivers/sensor.h>
#include <zephyr/drivers/sensor/vl53l0x.h>
#include <stdio.h>

#include "tof.h"

#define TOF_NODE DT_NODELABEL(tof)

// The control loop runs in main() at CONFIG_MAIN_THREAD_PRIORITY=0
#define TOF_THREAD_PRIORITY   5
#define TOF_THREAD_STACK_SIZE 2048

static struct tof_sample tof_shared;
static K_MUTEX_DEFINE(tof_lock);

void tof_get_latest(struct tof_sample *out)
{
	k_mutex_lock(&tof_lock, K_FOREVER);
	*out = tof_shared;
	k_mutex_unlock(&tof_lock);
}

bool tof_sample_fresh(const struct tof_sample *s)
{
	// timestamp 0 => nothing ever published, so age is the whole uptime.
	return s->timestamp != 0 &&
	       (k_uptime_get() - s->timestamp) <= TOF_SAMPLE_MAX_AGE_MS;
}

// This thread SLEEPS between status polls and never busy-waits or hogs the I2C bus
static void tof_thread_fn(void *p1, void *p2, void *p3)
{
	ARG_UNUSED(p1); ARG_UNUSED(p2); ARG_UNUSED(p3);

	const struct device *const tof = DEVICE_DT_GET(TOF_NODE);

	if (!device_is_ready(tof)) {
		printf("ERROR: ToF %s not ready -- height/vz stay at zero\n", tof->name);
		return;
	}

	// Previous sample, for the finite difference that produces vz.
	float prev_height = 0.0f;
	int64_t prev_time = 0;
	bool have_prev = false;

	while (1) {
		// One ranging measurement. Blocks ~33ms but yields (k_sleep) internally,
		// so the control loop keeps running throughout.
		int ret = sensor_sample_fetch(tof);
		if (ret) {
			printf("ToF fetch failed (%d)\n", ret);
			k_sleep(K_MSEC(10));
			continue;
		}

		// Fetch the distance and range status from the sensor. 
		// The range status is used to determine if the reading is valid.
		struct sensor_value distance, status;

		if (sensor_channel_get(tof, SENSOR_CHAN_DISTANCE, &distance) != 0) {
			k_sleep(K_MSEC(10));
			continue;
		}

		int range_status = -1;
		if (sensor_channel_get(tof,
				       (enum sensor_channel)SENSOR_CHAN_VL53L0X_RANGE_STATUS,
				       &status) == 0) {
			range_status = status.val1;
		}

		// Filter out invalid readings.
		if (range_status != VL53L0X_RANGE_STATUS_RANGE_VALID) {
			k_sleep(K_MSEC(5));
			continue;
		}

		// sensor_value is fixed point: val1 = whole metres, val2 = millionths.
		float height = (float)distance.val1 + (float)distance.val2 / 1000000.0f;
		int64_t now = k_uptime_get();

		// Get the vertical velocity (vz) by finite difference. 
		float vz = 0.0f;
		if (have_prev) {
			float dt = (float)(now - prev_time) / 1000.0f; 
			if (dt > 0.0f) {
				vz = (height - prev_height) / dt;
			}
		}

		// Update all three fields at once
		k_mutex_lock(&tof_lock, K_FOREVER);
		tof_shared.height    = height;
		tof_shared.vz        = vz;
		tof_shared.timestamp = now;
		k_mutex_unlock(&tof_lock);

		prev_height = height;
		prev_time   = now;
		have_prev   = true;

		// yield to the control loop and other threads.
		k_sleep(K_MSEC(5));  
	}
}

K_THREAD_DEFINE(tof_thread_id, TOF_THREAD_STACK_SIZE, tof_thread_fn,
		NULL, NULL, NULL, TOF_THREAD_PRIORITY, 0, 0);
