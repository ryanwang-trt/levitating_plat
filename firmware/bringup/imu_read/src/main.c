#include <zephyr/kernel.h>
#include <zephyr/device.h>
#include <zephyr/drivers/i2c.h>
#include <stdio.h>

#define I2C_NODE DT_NODELABEL(i2c21)

// Fixed control-loop rate. 200Hz -> 5ms period
// Matching the sensor's SMPLRT_DIV=4 output rate set in mpu6050_init.
#define CONTROL_HZ     200
#define CONTROL_PERIOD K_MSEC(1000 / CONTROL_HZ)

#define MPU6050_ADDR 0x68

// register map
#define REG_SMPLRT_DIV   0x19  // sample rate = base_rate / (1 + SMPLRT_DIV)
#define REG_CONFIG       0x1A  // bits2:0 DLPF_CFG (digital low-pass filter)
#define REG_PWR_MGMT_1   0x6B  // bit6 SLEEP; bits2:0 CLKSEL (clock source)
#define REG_GYRO_CONFIG  0x1B  // bits4:3 select gyro full-scale range
#define REG_ACCEL_CONFIG 0x1C  // bits4:3 select accel full-scale range
#define REG_ACCEL_XOUT_H 0x3B  // first of 14 big-endian bytes:
                               // AX AY AZ TEMP GX GY GZ (2 bytes each)

//Full-scale select values 
#define GYRO_FS_250DPS  (0 << 3)  // +/-250 deg/s
#define ACCEL_FS_2G     (0 << 3)  // +/-2 g

/* Scale factors for the ranges above (LSB per unit), from the datasheet:
     accel +/-2 g   -> 16384 LSB/g
     gyro  +/-250   -> 131.0 LSB/(deg/s)
   Temperature:  degC = raw/340 + 36.53 */
#define ACCEL_LSB_PER_G    16384.0f
#define GYRO_LSB_PER_DPS   131.0f

// I/O helpers (return 0 on success)
static int reg_write(const struct device *dev, uint8_t reg, uint8_t val)
{
	return i2c_reg_write_byte(dev, MPU6050_ADDR, reg, val);
}

static int burst_read(const struct device *dev, uint8_t start,
		      uint8_t *buf, uint32_t len)
{
	return i2c_burst_read(dev, MPU6050_ADDR, start, buf, len);
}

/* Combine a big-endian (high byte first) register pair into a signed 16-bit
   value. The MPU-6050 outputs are two's-complement, MSB at the lower address. */
static int16_t be16(const uint8_t *p)
{
	return (int16_t)((p[0] << 8) | p[1]);
}

//Configure the sensor. Return 0 if every write succeeded.
static int mpu6050_init(const struct device *dev)
{	
	//Wake up MPU6050 and clock it from the gyro-X PLL (CLKSEL=1), a more
	//stable timebase than the internal 8MHz oscillator (CLKSEL=0)
	int ret = reg_write(dev, REG_PWR_MGMT_1, 0x01);
	if (ret) {
		return ret; //return non 0 on fail writes
	}

	//DLPF_CFG=3: ~44Hz low-pass anti-aliasing filter. Also drops the gyro
	//base output rate to 1kHz, which SMPLRT_DIV below divides down
	ret = reg_write(dev, REG_CONFIG, 0x03);
	if (ret) {
		return ret;
	}

	//Sample rate = 1kHz / (1 + 4) = 200Hz. Depends on DLPF being enabled above
	ret = reg_write(dev, REG_SMPLRT_DIV, 0x04);
	if (ret) {
		return ret;
	}

	//Set gyro mearsure range (FS_SEL = 00, therefore low gain)
	ret = reg_write(dev, REG_GYRO_CONFIG, GYRO_FS_250DPS);
	if (ret) {
		return ret;
	}

	//Set accelometer range as above
	ret = reg_write(dev, REG_ACCEL_CONFIG, ACCEL_FS_2G);

	//Return 0 if all ret has value 0 (succeeded writes)
	return ret; 
}

//read one sample set and convert to physical units.
//write results through the out-pointers, return 0 on success.
static int read_sample(const struct device *dev,
		       float accel_g[3], float gyro_dps[3], float *temp_c)
{
	uint8_t raw[14];
	//reading reg contents into a raw array
	int ret = burst_read(dev,REG_ACCEL_XOUT_H,raw,sizeof(raw));
	if (ret) {
		return ret;
	}

	//combine higher and lower bits into 16 bits values
	int16_t ax=be16(&raw[0]);
	int16_t ay=be16(&raw[2]);
	int16_t az=be16(&raw[4]);

	int16_t temp=be16(&raw[6]);

	int16_t gyroX=be16(&raw[8]);
	int16_t gyroY=be16(&raw[10]);
	int16_t gyroZ=be16(&raw[12]);
	
	accel_g[0] = ax/ACCEL_LSB_PER_G;
	accel_g[1] = ay/ACCEL_LSB_PER_G;
	accel_g[2] = az/ACCEL_LSB_PER_G - 0.3f;  // clone Z zero-bias, from flip test

	gyro_dps[0] = gyroX/GYRO_LSB_PER_DPS;
	gyro_dps[1] = gyroY/GYRO_LSB_PER_DPS;
	gyro_dps[2] = gyroZ/GYRO_LSB_PER_DPS;

	*temp_c = temp/ 340.0f + 36.53f;
	
	return 0;
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

	if (mpu6050_init(i2c) != 0) {
		printf("ERROR: sensor init failed\n");
		return -EIO;
	}

	// Start the pacing timer: first tick one period from now, then every period.
	k_timer_start(&loop_timer, CONTROL_PERIOD, CONTROL_PERIOD);

	uint32_t tick = 0;

	while (1) {
		//buffers for accelometer, gyroscope and temperature datas
		float accel_g[3], gyro_dps[3], temp_c;

		uint32_t start = k_cycle_get_32();
		int rc = read_sample(i2c, accel_g, gyro_dps, &temp_c);
		uint32_t end = k_cycle_get_32();

		//Get the read duration for later engineering choices. (total time = 5ms)
		uint32_t read_time = k_cyc_to_us_floor32(end - start);

		//Per 100 ticks
		if (tick % 100 == 0) {
			if (rc == 0) {
				printf("t=%u up=%ums read_duration=%uus Az=% .2f\n",
				    tick, k_uptime_get_32(), read_time,
				    (double)accel_g[2]);
			} else {
				printf("t=%u read failed\n", tick);
			}
		}

		tick++;

		// Block until the next timer tick

		//total elapsed ticks in this thread (should be only 1) 
		uint32_t elapsed = k_timer_status_sync(&loop_timer);

		// if more than 1 tick has passed it means the work is too long for 5ms
		if (elapsed>1){
			uint32_t missed = elapsed - 1;
			printf("OVERRUN at t=%u: missed %u tick(s)\n", tick, missed);
		}
	}

	return 0;
}
