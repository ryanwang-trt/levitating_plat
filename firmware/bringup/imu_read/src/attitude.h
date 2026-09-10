// Complementary filter for roll/pitch.

#ifndef ATTITUDE_H
#define ATTITUDE_H

struct attitude {
	//filtered, rad
	float roll;       
	float pitch;        
	//raw accel branch, rad
	float roll_accel;  
	float pitch_accel;  
};

// Update the attitude estimate from the latest accel/gyro sample. 
void attitude_update(const float accel_g[3], const float gyro_dps[3],
		     float dt, struct attitude *out);

#endif  // ATTITUDE_H
