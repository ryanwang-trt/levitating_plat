#include <zephyr/kernel.h>
#include <zephyr/device.h>
#include <zephyr/drivers/i2c.h>
#include <stdio.h>

#define I2C_NODE DT_NODELABEL(i2c21)

#define MPU6050_ADDR 0x68

// register map
#define REG_PWR_MGMT_1   0x6B  // bit6 SLEEP=1 at power-on; write 0 to wake
#define REG_GYRO_CONFIG  0x1B  // bits4:3 select gyro full-scale range
#define REG_ACCEL_CONFIG 0x1C  // bits4:3 select accel full-scale range
#define REG_ACCEL_XOUT_H 0x3B  // first of 14 big-endian bytes:
                               // AX AY AZ TEMP GX GY GZ (2 bytes each)

/* Full-scale select values (write these into the *_CONFIG registers).
   We pick the most sensitive range for both. */
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
	//Wake up MPU6050
	int ret = reg_write(dev, REG_PWR_MGMT_1, 0x00);
	if (ret) {
		return ret; //return non 0 on fail writes
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

	while (1) {
		float accel_g[3], gyro_dps[3], temp_c;

		if (read_sample(i2c, accel_g, gyro_dps, &temp_c) == 0) {
			 printf("A[g] % .2f % .2f % .2f  "
				"G[dps] % .1f % .1f % .1f  T %.1fC\n",
			    accel_g[0], accel_g[1], accel_g[2],
				gyro_dps[0], gyro_dps[1], gyro_dps[2], temp_c);
		} else {
			printf("read failed\n");
		}

		// 10 Hz — slow enough to read on the terminal by eye.
		k_sleep(K_MSEC(100));
	}

	return 0;
}
