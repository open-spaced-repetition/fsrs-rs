use crate::{parameter_initialization::INIT_S_MAX, simulation::S_MIN};

const FSRS7_PARAM_LEN: usize = 35;

fn clamp_safe(value: f32, low: f32, high: f32) -> f32 {
    let low = if low.is_finite() { low } else { 0.0 };
    let high = if high.is_finite() { high } else { low };
    let (low, high) = if low <= high {
        (low, high)
    } else {
        (high, low)
    };
    let value = if value.is_finite() { value } else { low };
    value.clamp(low, high)
}

pub(crate) fn clip_fsrs7_parameters(parameters: &mut [f32]) {
    if parameters.len() < FSRS7_PARAM_LEN {
        return;
    }

    parameters[0] = clamp_safe(parameters[0], S_MIN, INIT_S_MAX / 2.0);
    parameters[1] = clamp_safe(parameters[1], parameters[0], INIT_S_MAX);
    parameters[2] = clamp_safe(parameters[2], parameters[1], INIT_S_MAX);
    parameters[3] = clamp_safe(parameters[3], parameters[2], INIT_S_MAX);

    parameters[4] = clamp_safe(parameters[4], 1.0, 10.0);
    parameters[5] = clamp_safe(parameters[5], 0.001, 4.0);
    parameters[6] = clamp_safe(parameters[6], 0.1, 4.0);

    parameters[7] = clamp_safe(parameters[7], 0.0, 4.0);
    parameters[8] = clamp_safe(parameters[8], 0.0, 1.2);
    parameters[9] = clamp_safe(parameters[9], 0.3, 3.0);
    parameters[10] = clamp_safe(parameters[10], 0.01, 1.5);
    parameters[11] = clamp_safe(parameters[11], 0.001, 0.9);
    parameters[12] = clamp_safe(parameters[12], 0.1, 1.0);
    parameters[13] = clamp_safe(parameters[13], 0.0, 3.5);
    parameters[14] = clamp_safe(parameters[14], 0.0, 1.0);
    parameters[15] = clamp_safe(parameters[15], 1.0, 7.0);

    parameters[16] = clamp_safe(parameters[16], 0.0, 4.0);
    parameters[17] = clamp_safe(parameters[17], 0.0, 2.0);
    parameters[18] = clamp_safe(parameters[18], 0.5, 6.0);
    parameters[19] = clamp_safe(parameters[19], 0.001, 1.5);
    parameters[20] = clamp_safe(parameters[20], 0.001, 2.0);
    parameters[21] = clamp_safe(parameters[21], 0.001, 1.0);
    parameters[22] = clamp_safe(parameters[22], 0.0, 5.0);
    parameters[23] = clamp_safe(parameters[23], 0.0, 1.0);
    parameters[24] = clamp_safe(parameters[24], 1.0, 7.0);

    parameters[25] = clamp_safe(parameters[25], 2.5, 15.0);
    parameters[26] = clamp_safe(parameters[26], 0.0, 1.0);

    parameters[27] = clamp_safe(parameters[27], 0.01, 0.25);
    parameters[28] = clamp_safe(parameters[28], parameters[27], 0.95);
    parameters[29] = clamp_safe(parameters[29], 0.5, 0.85);
    parameters[30] = clamp_safe(parameters[30], parameters[29], 0.99);
    parameters[31] = clamp_safe(parameters[31], 0.01, 1.0);
    parameters[32] = clamp_safe(parameters[32], 0.1, 1.0);
    parameters[33] = clamp_safe(parameters[33], 0.0, 0.9);
    parameters[34] = clamp_safe(parameters[34], 0.1, 1.1);
}
