pub static DEFAULT_PARAMETERS: [f32; 35] = [
    0.041, 2.4175, 4.1283, 11.9709, 5.6385, 0.4468, 3.262, 2.3054, 0.1688, 1.3325, 0.3524, 0.0049,
    0.7503, 0.0896, 0.6625, 1.3, 0.882, 0.3072, 3.5875, 0.303, 0.0107, 0.2279, 2.6413, 0.5594, 1.3,
    2.5, 1.0, 0.0723, 0.1634, 0.5, 0.9555, 0.2245, 0.6232, 0.1362, 0.3862,
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Result;
    use crate::inference::MemoryState;
    use crate::model::FSRS;

    #[test]
    fn test_fsrs7_retrievability_does_not_depend_on_w20() {
        let state = MemoryState {
            stability: 12.0,
            difficulty: 5.0,
        };
        let mut params_a = DEFAULT_PARAMETERS.to_vec();
        let mut params_b = DEFAULT_PARAMETERS.to_vec();
        params_a[20] = 0.001;
        params_b[20] = 5.0;
        let fsrs_a = FSRS::new(&params_a).unwrap();
        let fsrs_b = FSRS::new(&params_b).unwrap();
        let r_a = fsrs_a.current_retrievability(state, 10.0);
        let r_b = fsrs_b.current_retrievability(state, 10.0);
        assert!((r_a - r_b).abs() < 1e-7);
    }

    #[test]
    fn test_memory_from_sm2_fsrs7_bridge_is_finite() -> Result<()> {
        let params = vec![
            0.1558, 3.0107, 6.2423, 22.3570, 5.6837, 0.5279, 2.2999, 1.9751, 0.2886, 1.2884,
            0.8518, 0.0149, 0.7189, 0.6297, 0.3777, 2.8929, 0.9740, 0.5923, 3.6757, 0.8299, 0.0010,
            0.6994, 2.6457, 0.5673, 1.3138, 2.5067, 0.9955, 0.0499, 0.4071, 0.5686, 0.8969, 0.2210,
            0.8008, 0.0147, 0.1591,
        ];
        let fsrs = FSRS::new(&params)?;
        let state = fsrs.memory_state_from_sm2(2.5, 100.0, 0.9)?;
        assert!(state.stability.is_finite());
        assert!(state.stability > 0.0);
        assert!(state.difficulty.is_finite());
        assert!((state.difficulty - 5.0).abs() <= f32::EPSILON);
        Ok(())
    }

    #[test]
    fn test_s90_fsrs7_hits_90_retrievability() {
        let fsrs = FSRS::new(&DEFAULT_PARAMETERS).unwrap();
        let state = MemoryState {
            stability: 10.0,
            difficulty: 5.0,
        };

        let s90 = fsrs.s90(state);
        let r = fsrs.current_retrievability(state, s90);
        assert!((r - 0.9).abs() <= 1e-3);
    }
}
