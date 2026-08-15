#include "policy.h"
#include "policy_weights.h"
#include <math.h>
#include <stddef.h>

static inline float clampf(float value, float lo, float hi)
{
	if (value < lo) {
		return lo;
	}
	if (value > hi) {
		return hi;
	}
	return value;
}

// One dense layer with tanh activation. Each of the num_neurons neurons takes a
// weighted sum of all num_inputs inputs (plus its own bias), then squashes it:
static void dense_tanh(const float *weights, const float *biases,
		       const float *input, float *output,
		       int num_neurons, int num_inputs)
{
	for (int neuron = 0; neuron < num_neurons; neuron++) {
		float sum = biases[neuron];
		
		const float *neuron_weights = weights + (size_t)neuron * num_inputs;
		for (int input_idx = 0; input_idx < num_inputs; input_idx++) {
			sum += neuron_weights[input_idx] * input[input_idx];
		}
		output[neuron] = tanhf(sum);
	}
}

void policy_forward(const float obs[6], float action[4])
{
	float hidden1[POLICY_H1];
	float hidden2[POLICY_H2];

	// Two tanh hidden layers: obs -> hidden1 -> hidden2.
	dense_tanh(&POLICY_W0[0][0], POLICY_B0, obs, hidden1, POLICY_H1, POLICY_OBS_DIM);
	dense_tanh(&POLICY_W2[0][0], POLICY_B2, hidden1, hidden2, POLICY_H2, POLICY_H1);

	// Output layer: same weighted sum, but clamp to the [0,1] motor range
	// instead of tanh (each output is one motor command).
	for (int motor = 0; motor < POLICY_ACT_DIM; motor++) {
		float sum = POLICY_BA[motor];
		const float *motor_weights = &POLICY_WA[motor][0];
		for (int input_idx = 0; input_idx < POLICY_H2; input_idx++) {
			sum += motor_weights[input_idx] * hidden2[input_idx];
		}
		action[motor] = clampf(sum, 0.0f, 1.0f);
	}
}

int policy_self_check(float tolerance, float *max_error)
{
	float computed_action[POLICY_ACT_DIM];
	policy_forward(POLICY_REF_OBS, computed_action);

	// Largest absolute error across the 4 outputs vs the compiled-in reference.
	float worst_error = 0.0f;
	for (int i = 0; i < POLICY_ACT_DIM; i++) {
		float abs_error = fabsf(computed_action[i] - POLICY_REF_ACTION[i]);
		if (abs_error > worst_error) {
			worst_error = abs_error;
		}
	}

	if (max_error != NULL) {
		*max_error = worst_error;
	}
	return (worst_error <= tolerance) ? 0 : -1;
}
