use crate::{
    parameter_initialization::INIT_S_MAX,
    simulation::{D_MAX, D_MIN, S_MIN},
};

pub(crate) fn clip_fsrs6_parameters(
    parameters: &mut [f32],
    num_relearning_steps: usize,
    enable_short_term: bool,
) {
    let w17_w18_ceiling = if parameters.len() > 14 && num_relearning_steps > 1 {
        (-(parameters[11].ln() + (2.0f32.powf(parameters[13]) - 1.0).ln() + parameters[14] * 0.3)
            / num_relearning_steps as f32)
            .max(0.01)
            .sqrt()
            .min(2.0)
    } else {
        2.0
    };
    let w19_floor = if enable_short_term { 0.01 } else { 0.0 };

    let clamps: [(f32, f32); 21] = [
        (S_MIN, INIT_S_MAX),
        (S_MIN, INIT_S_MAX),
        (S_MIN, INIT_S_MAX),
        (S_MIN, INIT_S_MAX),
        (D_MIN, D_MAX),
        (0.001, 4.0),
        (0.001, 4.0),
        (0.001, 0.75),
        (0.0, 4.5),
        (0.0, 0.8),
        (0.001, 3.5),
        (0.001, 5.0),
        (0.001, 0.25),
        (0.001, 0.9),
        (0.0, 4.0),
        (0.0, 1.0),
        (1.0, 6.0),
        (0.0, w17_w18_ceiling),
        (0.0, w17_w18_ceiling),
        (w19_floor, 0.8),
        (0.1, 0.8),
    ];

    parameters
        .iter_mut()
        .zip(clamps)
        .for_each(|(w, (low, high))| *w = w.clamp(low, high));
}
