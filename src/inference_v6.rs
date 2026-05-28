pub const FSRS5_DEFAULT_DECAY: f32 = 0.5;
pub const FSRS6_DEFAULT_DECAY: f32 = 0.1542;

pub static FSRS6_DEFAULT_PARAMETERS: [f32; 21] = [
    0.212,
    1.2931,
    2.3065,
    8.2956,
    6.4133,
    0.8334,
    3.0194,
    0.001,
    1.8722,
    0.1666,
    0.796,
    1.4835,
    0.0614,
    0.2629,
    1.6483,
    0.6014,
    1.8729,
    0.5425,
    0.0912,
    0.0658,
    FSRS6_DEFAULT_DECAY,
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::{MemoryState, current_retrievability};
    use crate::model::FSRS;

    #[test]
    fn test_model_current_retrievability_matches_fsrs6_scalar() {
        let fsrs = FSRS::new(&FSRS6_DEFAULT_PARAMETERS).unwrap();
        let state = MemoryState {
            stability: 10.0,
            difficulty: 5.0,
        };
        let expected = current_retrievability(state, 3.0, FSRS6_DEFAULT_DECAY);
        let actual = fsrs.current_retrievability(state, 3.0);
        assert!((actual - expected).abs() < 1e-6);
    }

    #[test]
    fn test_s90_legacy_equals_stability() {
        let fsrs = FSRS::new(&FSRS6_DEFAULT_PARAMETERS).unwrap();
        let state = MemoryState {
            stability: 12.345,
            difficulty: 5.0,
        };

        assert_eq!(fsrs.s90(state), state.stability);
    }
}
