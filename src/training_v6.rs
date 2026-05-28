use super::training_v7;

const PARAMS_STDDEV: [f32; 21] = [
    6.43, 9.66, 17.58, 27.85, 0.57, 0.28, 0.6, 0.12, 0.39, 0.18, 0.33, 0.3, 0.09, 0.16, 0.57, 0.25,
    1.03, 0.31, 0.32, 0.14, 0.27,
];

pub(crate) fn l2_penalty_value_and_grad(
    w: &[f32],
    init_w: &[f32],
    batch_size: usize,
    total_size: usize,
    l2_weight: f64,
    _params_stddev: &[f32],
) -> (f64, Vec<f32>) {
    let mut grad = vec![0.0f32; w.len()];
    if total_size == 0 {
        return (0.0, grad);
    }
    let size = w.len().min(init_w.len()).min(PARAMS_STDDEV.len());
    let scale = l2_weight * batch_size as f64 / total_size as f64;
    let mut penalty_sum = 0.0f64;
    for i in 0..size {
        let sigma = PARAMS_STDDEV[i] as f64;
        let denom = sigma * sigma;
        let diff = w[i] as f64 - init_w[i] as f64;
        penalty_sum += diff * diff / denom;
        grad[i] = (2.0 * diff / denom * scale) as f32;
    }
    let penalty = penalty_sum * scale;
    if !penalty.is_finite() {
        return (0.0, vec![0.0; w.len()]);
    }
    for g in &mut grad {
        if !g.is_finite() {
            *g = 0.0;
        }
    }
    (penalty, grad)
}

pub(crate) fn maybe_schedule_penalty_value_and_grad(
    _w: &[f32],
    _batch_size: usize,
    _enable_sched_penalties: bool,
) -> (f64, [f64; training_v7::GRAD_LEN]) {
    (0.0, [0.0; training_v7::GRAD_LEN])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fsrs6_schedule_penalty_is_noop() {
        let w = crate::inference::FSRS6_DEFAULT_PARAMETERS.to_vec();
        let (value, grad) = maybe_schedule_penalty_value_and_grad(&w, 512, true);
        assert_eq!(value, 0.0);
        assert!(grad.iter().all(|g| *g == 0.0));
    }

    #[test]
    fn test_fsrs6_l2_penalty_is_zero_for_initial_params() {
        let w = crate::inference::FSRS6_DEFAULT_PARAMETERS.to_vec();
        let init_w = w.clone();
        let (value, grad) = l2_penalty_value_and_grad(
            &w,
            &init_w,
            512,
            1000,
            training_v7::PENALTY_W_L2,
            &training_v7::PARAMS_STDDEV,
        );
        assert_eq!(value, 0.0);
        assert_eq!(grad.len(), w.len());
        assert!(grad.iter().all(|g| *g == 0.0));
    }

    #[test]
    fn test_fsrs6_l2_penalty_applies_to_changed_params() {
        let init_w = crate::inference::FSRS6_DEFAULT_PARAMETERS.to_vec();
        let mut w = init_w.clone();
        w[4] += 0.57;
        let (value, grad) = l2_penalty_value_and_grad(
            &w,
            &init_w,
            512,
            1000,
            training_v7::PENALTY_W_L2,
            &training_v7::PARAMS_STDDEV,
        );
        assert!((value - 0.256).abs() < 1e-6);
        assert_eq!(grad.len(), w.len());
        assert!((grad[4] - 0.8982456).abs() < 1e-6);
        assert!(
            grad.iter()
                .enumerate()
                .all(|(idx, grad)| idx == 4 || *grad == 0.0)
        );
    }
}
