#include <zephyr/kernel.h>
#include <zephyr/device.h>
#include <zephyr/drivers/sensor.h>
#include <zephyr/drivers/sensor/vl53l0x.h>   // vendor channels + RANGE_STATUS_* values
#include <stdio.h>

// Matches the `tof:` label on the vl53l0x@29 node in the board overlay.
#define TOF_NODE DT_NODELABEL(tof)

int main(void)
{
	const struct device *const tof = DEVICE_DT_GET(TOF_NODE);

	printf("\n=== tof_test: VL53L0X on %s ===\n", tof->name);

	// Driver init runs before main
	if (!device_is_ready(tof)) {
		printf("ERROR: %s not ready -- check wiring/address\n", tof->name);
		return -ENODEV;
	}

	while (1) {
		// One measurement, BLOCKS ~33ms 
		int ret = sensor_sample_fetch(tof);

		// return 0 on success 
		if (ret) {
			printf("fetch failed (%d)\n", ret);
			k_sleep(K_MSEC(100));
			continue;
		}

		struct sensor_value distance, status;

		ret = sensor_channel_get(tof, SENSOR_CHAN_DISTANCE, &distance);
		if (ret) {
			printf("distance read failed (%d)\n", ret);
			k_sleep(K_MSEC(100));
			continue;
		}

		// val1 = whole metres, val2 = millionths.
		int distance_mm = distance.val1 * 1000 + distance.val2 / 1000;

		// Range status: only status 0 (RANGE_VALID) is trustworthy. 
		int range_status = -1;
		if (sensor_channel_get(tof,
				       (enum sensor_channel)SENSOR_CHAN_VL53L0X_RANGE_STATUS,
				       &status) == 0) {
			range_status = status.val1;
		}

		printf("distance = %5d mm   status = %d %s\n",
		       distance_mm, range_status,
		       range_status == VL53L0X_RANGE_STATUS_RANGE_VALID
			       ? "(valid)" : "(INVALID - do not trust)");

		// The fetch already costs ~33ms; this just paces prints so the console
		// is readable while you hold a tape measure to it.
		k_sleep(K_MSEC(100));
	}

	return 0;
}
