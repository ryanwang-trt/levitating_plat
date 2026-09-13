#include <zephyr/kernel.h>
#include <zephyr/device.h>
#include <stdio.h>
#include <math.h>

#include "policy.h"
#include "imu.h"
#include "tof.h"
#include "attitude.h"

#define I2C_NODE DT_NODELABEL(i2c21)

// Fixed control-loop rate. 200Hz -> 5ms period
// Matching the sensor's SMPLRT_DIV=4 output rate set in imu_init.
#define CONTROL_HZ 200
#define CONTROL_PERIOD K_MSEC(1000 / CONTROL_HZ)

// Timing stats window: 400 ticks = 2s at 200Hz.
#define WINDOW_TICKS 400

#define TARGET_HEIGHT_M 0.5f   // sim hover target (env.target_height)
#define DEG2RAD (3.14159265f / 180.0f)

// Assemble the 6-element observation the policy expects, in the SAME units as the
// training sim (env._get_obs):
//   obs[0] height (m)         obs[1] vz (m/s)
//   obs[2] roll (rad)         obs[3] pitch (rad)
//   obs[4] roll_rate (rad/s)  obs[5] pitch_rate (rad/s)
static void build_obs(const struct attitude *att, const struct tof_sample *tof,
		      const float gyro_dps[3], float obs[6])
{

	if (tof_sample_fresh(tof)) {
		obs[0] = tof->height;
		obs[1] = tof->vz;
	} else {
		obs[0] = TARGET_HEIGHT_M;
		obs[1] = 0.0f;
	}

	obs[2] = att->roll;
	obs[3] = att->pitch;

	// Caliberated gyro rates
	obs[4] = gyro_dps[0] * DEG2RAD;
	obs[5] = gyro_dps[1] * DEG2RAD;
}


// Periodic pacing timer for the control loop. No expiry/stop callbacks —
// loop just waits on its ticks via k_timer_status_sync().
K_TIMER_DEFINE(loop_timer, NULL, NULL);

int main(void)
{
	const struct device *const i2c = DEVICE_DT_GET(I2C_NODE);

	printf("\n=== imu_read: MPU-6050 on %s ===\n", i2c->name);

	if (!device_is_ready(i2c)) {
		printf("ERROR: %s not ready\n", i2c->name);
		return -ENODEV;
	}

	if (imu_init(i2c) != 0) {
		printf("ERROR: sensor init failed\n");
		return -EIO;
	}

	// Verify the flashed policy reproduces its reference before the loop uses it.
	float selfcheck_error;
	if (policy_self_check(1e-3f, &selfcheck_error) == 0) {
		printf("policy self-check PASS (max err %.2e)\n", (double)selfcheck_error);
	} else {
		printf("policy self-check FAIL (err %.2e): output != reference; "
		       "policy_weights.h corrupt or build broken\n", (double)selfcheck_error);
	}

	printf("T+/T- = ToF on/off, rd min/mean/max, o=overruns | H = height, V = vz filt/raw, a = age\n");

	// Measure sensor bias before flying. Board must sit still and level.
	printf("calibrating -- hold still & level...\n");
	imu_calibrate(i2c);

	// Start the pacing timer: first tick one period from now, then every period.
	k_timer_start(&loop_timer, CONTROL_PERIOD, CONTROL_PERIOD);

	uint32_t tick = 0;

	// Starts true so the first stale sample prints a warning
	bool tof_was_ok = true;

	// Timing stats window state.
	uint32_t win_rd_min = UINT32_MAX, win_rd_max = 0, win_rd_sum = 0;
	uint32_t win_inf_min = UINT32_MAX, win_inf_max = 0, win_inf_sum = 0;
	uint32_t win_ticks = 0;
	uint32_t overrun_total = 0;
	bool win_tof_on = true;

	while (1) {
		//buffer for accelometer, gyroscope and temperature datas
		struct imu_sample imu;

		uint32_t start = k_cycle_get_32();
		int rc = imu_read(i2c, &imu);
		uint32_t end = k_cycle_get_32();

		//Get the read duration for later engineering choices. (total time = 5ms)
		uint32_t read_time = k_cyc_to_us_floor32(end - start);

		// Snapshot the latest ToF sample (wont block main loop)
		struct tof_sample tof;
		tof_get_latest(&tof);

		// Run the complementary filter every tick
		struct attitude att;
		attitude_update(imu.accel_g, imu.gyro_dps, 1.0f / CONTROL_HZ, &att);

		// edge detection for when tof sample is STALE or RECOVERED
		// print the message only when the state changes 
		bool tof_ok = tof_sample_fresh(&tof);
		tof_was_ok = tof_ok;   // transition print silenced while testing filters

		// assemble the obs, run the policy, time the inference.
		float obs[6], action[4];
		build_obs(&att, &tof, imu.gyro_dps, obs);

		uint32_t infer_start = k_cycle_get_32();
		policy_forward(obs, action);
		uint32_t infer_time = k_cyc_to_us_floor32(k_cycle_get_32() - infer_start);

		// Timing stats accumulated on EVERY tick
		if (read_time < win_rd_min) win_rd_min = read_time;
		if (read_time > win_rd_max) win_rd_max = read_time;
		win_rd_sum += read_time;
		if (infer_time < win_inf_min) win_inf_min = infer_time;
		if (infer_time > win_inf_max) win_inf_max = infer_time;
		win_inf_sum += infer_time;
		win_ticks++;

		// Every WINDOW_TICKS: report, then flip the ToF on/off. 
		if (win_ticks >= WINDOW_TICKS) {
			// Short on purpose: a long line blocks past the 5ms tick and would
			// cause the very overruns this is counting.
			printf("T%c %u/%u/%u o%u\n",
			       win_tof_on ? '+' : '-',
			       win_rd_min, win_rd_sum / win_ticks, win_rd_max,
			       overrun_total);

			// A/B toggle off: ToF stays on so age reflects its real update rate.

			win_rd_min = UINT32_MAX; win_rd_max = 0; win_rd_sum = 0;
			win_inf_min = UINT32_MAX; win_inf_max = 0; win_inf_sum = 0;
			win_ticks = 0;
		}

		//print some debug info every 100 ticks (0.5s) alternating between attitude and ToF
		if (tick % 100 == 0) {
			if (rc != 0) {
				printf("t=%u read failed\n", tick);
			} else {
				printf("H %.3f V %+.2f/%+.2f a%d\n",
				       (double)tof.height,
				       (double)tof.vz, (double)tof.vz_raw,
				       (int)(k_uptime_get() - tof.timestamp));
			}
		}

		tick++;

		// Block until the next timer tick

		//total elapsed ticks in this thread (should be only 1)
		uint32_t elapsed = k_timer_status_sync(&loop_timer);

		// if more than 1 tick has passed it means the work is too long for 5ms
		if (elapsed > 1) {
			uint32_t missed = elapsed - 1;
			overrun_total += missed;
			printf("OVERRUN at t=%u: missed %u tick(s)\n", tick, missed);
		}
	}

	return 0;
}
