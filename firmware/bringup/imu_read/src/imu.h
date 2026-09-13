#ifndef IMU_H
#define IMU_H

#include <zephyr/device.h>

// One sample set, scaled to physical units with the boot bias subtracted.
struct imu_sample {
	float accel_g[3];    // accel X/Y/Z in g
	float gyro_dps[3];   // gyro X/Y/Z in deg/s
	float temp_c;        // die temperature in degC
};

// Configure the sensor. Return 0 if every write succeeded.
int imu_init(const struct device *dev);

// Read one sample set and convert to physical units.
// Write results through the out-pointer, return 0 on success.
int imu_read(const struct device *dev, struct imu_sample *out);

// Measure per-sensor bias by averaging N samples while the board is held STILL
// and LEVEL to subtract from future mearsurement. Blocks ~1s.
void imu_calibrate(const struct device *dev);

#endif // IMU_H
