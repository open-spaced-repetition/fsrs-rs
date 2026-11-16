pub static DEFAULT_PARAMETERS: [f32; 34] = [
    0.1104, 2.2395, 3.9221, 11.7841, 6.1686, 0.6457, 3.6807, 1.9795, 0.0, 1.3826, 0.7024, 0.5999,
    0.8146, 0.6398, 1.0, 1.3207, 0.6707, 3.8668, 0.4416, 0.0934, 1.8631, 0.6162, 1.0869, 0.1567,
    0.0801, 0.2421, 0.9464, 0.1433, 0.7145, 0.0, 0.5667, 0.3734, 0.5333, 0.3048,
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::{FSRSError, Result};
    use crate::inference::MemoryState;
    use crate::model::FSRS;

    #[test]
    fn test_fsrs7_retrievability_does_not_depend_on_w20() {
        let state = MemoryState {
            stability: 12.0,
            difficulty: 5.0,
            stability_fast: 9.6,
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
    fn test_memory_from_sm2_fsrs7_preserves_interval_at_retention() -> Result<()> {
        let mut alternate_parameters = DEFAULT_PARAMETERS;
        alternate_parameters[23] = 0.3;
        alternate_parameters[27] = 0.7;
        alternate_parameters[29] = 0.2;
        alternate_parameters[30] = 0.8;

        for parameters in [DEFAULT_PARAMETERS, alternate_parameters] {
            let fsrs = FSRS::new(&parameters)?;
            for retention in [0.8, 0.9, 0.95] {
                for expected_interval in [1.0, 30.0, 100.0] {
                    let state = fsrs.memory_state_from_sm2(2.5, expected_interval, retention)?;
                    let interval = fsrs.interval_at_retrievability(state, retention);
                    assert!(state.stability.is_finite());
                    assert!(state.stability > 0.0);
                    assert!(state.stability_fast.is_finite());
                    assert!(state.stability_fast > 0.0);
                    assert!((state.stability_fast / state.stability - 0.8).abs() < 1e-6);
                    assert!((state.difficulty - 5.0).abs() <= f32::EPSILON);
                    assert!(
                        (interval - expected_interval).abs() < 0.01,
                        "retention={retention}, expected={expected_interval}, actual={interval}"
                    );
                }
            }
        }
        Ok(())
    }

    fn alternate_parameters() -> [f32; 34] {
        let mut parameters = DEFAULT_PARAMETERS;
        parameters[23] = 0.3;
        parameters[27] = 0.7;
        parameters[29] = 0.2;
        parameters[30] = 0.8;
        parameters
    }

    #[test]
    fn test_scale_memory_state_fsrs7_reaches_retention_at_interval() -> Result<()> {
        for parameters in [DEFAULT_PARAMETERS, alternate_parameters()] {
            let fsrs = FSRS::new(&parameters)?;
            for difficulty in [1.0, 5.0, 9.5] {
                for ratio in [0.3, 0.8, 1.5] {
                    for retention in [0.8, 0.9, 0.95] {
                        for target in [1.0f32, 30.0, 100.0] {
                            let shape = MemoryState {
                                stability: 7.0,
                                difficulty,
                                stability_fast: 7.0 * ratio,
                            };
                            let state =
                                fsrs.scale_memory_state_to_interval(shape, target, retention)?;
                            let interval = fsrs.interval_at_retrievability(state, retention);
                            let context = format!(
                                "d={difficulty}, ratio={ratio}, retention={retention}, target={target}"
                            );
                            assert_eq!(state.difficulty, difficulty, "{context}");
                            assert!(
                                (state.stability_fast / state.stability / ratio - 1.0).abs() < 1e-5,
                                "{context}"
                            );
                            let tolerance = 0.01f32.max(target * 1e-3);
                            assert!(
                                (interval - target).abs() < tolerance,
                                "{context}: {interval}"
                            );
                        }
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn test_scale_memory_state_matches_sm2_bridge_for_neutral_shape() -> Result<()> {
        let fsrs = FSRS::new(&DEFAULT_PARAMETERS)?;
        let shape = MemoryState {
            stability: 1.0,
            difficulty: 5.0,
            stability_fast: 0.8,
        };
        for retention in [0.8, 0.9, 0.95] {
            for interval in [1.0, 30.0, 100.0] {
                assert_eq!(
                    fsrs.scale_memory_state_to_interval(shape, interval, retention)?,
                    fsrs.memory_state_from_sm2(2.5, interval, retention)?
                );
            }
        }
        Ok(())
    }

    #[test]
    fn test_scale_memory_state_fsrs6_single_trace() -> Result<()> {
        let fsrs = FSRS::new(&crate::inference::FSRS6_DEFAULT_PARAMETERS)?;
        let shape = MemoryState {
            stability: 3.0,
            difficulty: 6.0,
            stability_fast: 1.0,
        };
        for retention in [0.8, 0.9, 0.95] {
            for target in [1.0, 30.0, 100.0] {
                let state = fsrs.scale_memory_state_to_interval(shape, target, retention)?;
                assert_eq!(state.stability_fast, state.stability);
                assert_eq!(state.difficulty, 6.0);
                let interval = fsrs.interval_at_retrievability(state, retention);
                assert!(
                    (interval - target).abs() < 0.01 * target.max(1.0),
                    "{interval}"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn test_scale_memory_state_rejects_invalid_input() -> Result<()> {
        let valid = MemoryState {
            stability: 5.0,
            difficulty: 5.0,
            stability_fast: 4.0,
        };
        for parameters in [
            DEFAULT_PARAMETERS.to_vec(),
            crate::inference::FSRS6_DEFAULT_PARAMETERS.to_vec(),
        ] {
            let fsrs = FSRS::new(&parameters)?;
            let call = |state: MemoryState, interval: f32, retention: f32| {
                fsrs.scale_memory_state_to_interval(state, interval, retention)
            };
            assert!(call(valid, 10.0, 0.9).is_ok());
            assert_eq!(call(valid, f32::NAN, 0.9), Err(FSRSError::InvalidInput));
            assert_eq!(call(valid, 10.0, 0.0), Err(FSRSError::InvalidInput));
            assert_eq!(call(valid, 10.0, 1.0), Err(FSRSError::InvalidInput));
            assert_eq!(call(valid, 10.0, f32::NAN), Err(FSRSError::InvalidInput));
            let zero_stability = MemoryState {
                stability: 0.0,
                ..valid
            };
            assert_eq!(
                call(zero_stability, 10.0, 0.9),
                Err(FSRSError::InvalidInput)
            );
            let negative_fast = MemoryState {
                stability_fast: -1.0,
                ..valid
            };
            assert_eq!(call(negative_fast, 10.0, 0.9), Err(FSRSError::InvalidInput));
            let nan_difficulty = MemoryState {
                difficulty: f32::NAN,
                ..valid
            };
            assert_eq!(
                call(nan_difficulty, 10.0, 0.9),
                Err(FSRSError::InvalidInput)
            );
        }
        Ok(())
    }

    #[test]
    fn test_scale_memory_state_out_of_range_returns_bound_state() -> Result<()> {
        let fsrs = FSRS::new(&DEFAULT_PARAMETERS)?;
        let shape = MemoryState {
            stability: 1.0,
            difficulty: 5.0,
            stability_fast: 0.8,
        };
        let huge = fsrs.scale_memory_state_to_interval(shape, 1e9, 0.95)?;
        assert_eq!(huge.stability, crate::simulation::S_MAX);
        let tiny = fsrs.scale_memory_state_to_interval(shape, 1e6, 1e-6)?;
        assert_eq!(tiny.stability, crate::simulation::S_MIN);
        Ok(())
    }

    #[test]
    fn test_s90_fsrs7_hits_90_retrievability() {
        let fsrs = FSRS::new(&DEFAULT_PARAMETERS).unwrap();
        let state = MemoryState {
            stability: 10.0,
            difficulty: 5.0,
            stability_fast: 8.0,
        };

        let s90 = fsrs.s90(state);
        let r = fsrs.current_retrievability(state, s90);
        assert!((r - 0.9).abs() <= 1e-3);
    }
}
