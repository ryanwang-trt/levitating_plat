/* Bring-up: raw-I2C scan of i2c21 (SDA=P1.03, SCL=P1.04) looking for the
   MPU-6050 at 0x68. Upstream Zephyr i2c.h only — no in-tree MPU-6050 driver.

   Probing uses a zero-length write, same as Zephyr's own `i2c scan` shell
   command: the transfer succeeds iff the target ACKs its address byte. */

#include <zephyr/kernel.h>
#include <zephyr/device.h>
#include <zephyr/drivers/i2c.h>
#include <zephyr/sys/printk.h>

#define I2C_NODE DT_NODELABEL(i2c21)

#define MPU6050_ADDR         0x68
#define MPU6050_REG_WHO_AM_I 0x75
#define MPU6050_WHO_AM_I_VAL 0x68

/* I2C reserves 0x00-0x07 and 0x78-0x7F; we still probe them because a hit
there means something is wrong, not that a second sensor showed up.*/
#define ADDR_RESERVED(a) ((a) <= 0x07 || (a) >= 0x78)

static bool probe(const struct device *dev, uint8_t addr)
{
	uint8_t dummy = 0;
	struct i2c_msg msg = {
		.buf = &dummy,	// must be a real pointer even at len 0
		.len = 0U,
		.flags = I2C_MSG_WRITE | I2C_MSG_STOP,
	};

	return i2c_transfer(dev, &msg, 1, addr) == 0;
}

static int scan_bus(const struct device *dev)
{
	int found = 0;

	printk("     0  1  2  3  4  5  6  7  8  9  a  b  c  d  e  f\n");

	for (uint8_t row = 0x00; row < 0x80; row += 0x10) {
		printk("%02x: ", row);

		for (uint8_t col = 0; col < 0x10; col++) {
			uint8_t addr = row + col;

			if (probe(dev, addr)) {
				printk("%02x ", addr);
				found++;
			} else {
				printk("-- ");
			}
		}

		printk("\n");
	}

	printk("%d device(s) found on %s\n", found, dev->name);

	return found;
}

static int read_who_am_i(const struct device *dev, uint8_t *val)
{
	return i2c_reg_read_byte(dev, MPU6050_ADDR, MPU6050_REG_WHO_AM_I, val);
}

int main(void)
{
	const struct device *const i2c = DEVICE_DT_GET(I2C_NODE);
	uint8_t who;
	int err;

	printk("\n=== i2c_scan: %s (SDA=P1.03, SCL=P1.04) ===\n", i2c->name);

	if (!device_is_ready(i2c)) {
		return -ENODEV;
	}

	scan_bus(i2c);

	// The MPU-6050 powers up in sleep mode, but WHO_AM_I is readable in sleep
	err = read_who_am_i(i2c, &who);
	if (err) {
		printk("WHO_AM_I read from 0x%02x failed (err %d)\n",
		       MPU6050_ADDR, err);
	} else if (who == MPU6050_WHO_AM_I_VAL) {
		printk("WHO_AM_I (0x%02x) = 0x%02x — MPU-6050 confirmed\n",
		       MPU6050_REG_WHO_AM_I, who);
	} else {
		printk("WHO_AM_I (0x%02x) = 0x%02x — expected 0x%02x; likely "
		       "an MPU-6050 clone\n",
		       MPU6050_REG_WHO_AM_I, who, MPU6050_WHO_AM_I_VAL);
	}

	// Keep polling so the output survives attaching the terminal late, and
	// so a wire coming loose shows up as reads starting to fail.
	while (1) {
		err = read_who_am_i(i2c, &who);
		if (err) {
			printk("WHO_AM_I err %d\n", err);
		} else {
			printk("WHO_AM_I ok (0x%02x)\n", who);
		}

		k_sleep(K_SECONDS(1));
	}

	return 0;
}
