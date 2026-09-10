#include <math.h>
#include <stdbool.h>

#include "attitude.h"

#define DEG2RAD (3.14159265f / 180.0f)

#define ATTITUDE_ALPHA 0.98f

static float roll_state;
static float pitch_state;
static bool initialized;

void attitude_update(const float accel_g[3], const float gyro_dps[3],
		     float dt, struct attitude *out)
{
	float ax = accel_g[0];
	float ay = accel_g[1];
	float az = accel_g[2];

	// accel branch, calculate roll/pitch from the gravity vector.
	float roll_accel = atan2f(ay, az);
	float pitch_accel = atan2f(-ax, sqrtf(ay * ay + az * az));

	if (!initialized) {
		roll_state = roll_accel;
		pitch_state = pitch_accel;
		initialized = true;
	} else {
		// Gyro branch, integrate the angular rate over timeto get the angle.
		float roll_gyro = gyro_dps[0] * DEG2RAD;
		float pitch_gyro = gyro_dps[1] * DEG2RAD;

		float alpha = ATTITUDE_ALPHA;

		roll_state = alpha * (roll_state + roll_gyro * dt) + (1.0f - alpha) * roll_accel;
		pitch_state = alpha * (pitch_state + pitch_gyro * dt) + (1.0f - alpha) * pitch_accel;
	}

	out->roll = roll_state;
	out->pitch = pitch_state;
	out->roll_accel = roll_accel;
	out->pitch_accel = pitch_accel;
}
