use crate::DEFAULT_PARAMETERS;
use crate::error::{FSRSError, Result};
use crate::simulation::S_MIN;
use std::collections::HashMap;

static R_S0_DEFAULT_ARRAY: &[(u32, f32); 4] = &[
    (1, DEFAULT_PARAMETERS[0]),
    (2, DEFAULT_PARAMETERS[1]),
    (3, DEFAULT_PARAMETERS[2]),
    (4, DEFAULT_PARAMETERS[3]),
];

const INIT_S_MAX: f32 = 100.0;

fn fill_initial_stabilities_fsrs7(rating_stability: &HashMap<u32, f32>) -> Result<[f32; 4]> {
    if rating_stability.is_empty() {
        return Err(FSRSError::NotEnoughData);
    }

    let default_s0 = R_S0_DEFAULT_ARRAY
        .iter()
        .cloned()
        .collect::<HashMap<_, _>>();

    if rating_stability.len() == 1 {
        let rating = rating_stability.keys().next().copied().unwrap();
        let factor = rating_stability[&rating] / default_s0[&rating];
        let mut values = [
            default_s0[&1] * factor,
            default_s0[&2] * factor,
            default_s0[&3] * factor,
            default_s0[&4] * factor,
        ];
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        return Ok(values.map(|v| v.clamp(S_MIN, INIT_S_MAX)));
    }

    let anchors: HashMap<u32, f64> = HashMap::from([(1, -8.09), (2, -3.83), (3, -2.5), (4, -1.0)]);
    let mut log_s0: HashMap<u32, f64> = rating_stability
        .iter()
        .map(|(k, v)| (*k, (*v as f64).ln()))
        .collect();

    for target in 1..=4 {
        if log_s0.contains_key(&target) {
            continue;
        }
        let lower = (1..target).rev().find(|r| log_s0.contains_key(r));
        let upper = ((target + 1)..=4).find(|r| log_s0.contains_key(r));

        let value = match (lower, upper) {
            (Some(lo), Some(hi)) => {
                let t = (anchors[&target] - anchors[&lo]) / (anchors[&hi] - anchors[&lo]);
                log_s0[&lo] + t * (log_s0[&hi] - log_s0[&lo])
            }
            (Some(lo), None) => log_s0[&lo] + (anchors[&target] - anchors[&lo]),
            (None, Some(hi)) => log_s0[&hi] + (anchors[&target] - anchors[&hi]),
            (None, None) => return Err(FSRSError::NotEnoughData),
        };
        log_s0.insert(target, value);
    }

    let mut values = [
        log_s0[&1].exp() as f32,
        log_s0[&2].exp() as f32,
        log_s0[&3].exp() as f32,
        log_s0[&4].exp() as f32,
    ];
    for value in &mut values {
        *value = value.clamp(0.0001, INIT_S_MAX);
    }
    for i in 1..values.len() {
        values[i] = values[i].max(values[i - 1]);
    }

    Ok(values.map(|v| v.clamp(S_MIN, INIT_S_MAX)))
}

pub(crate) fn smooth_initial_stabilities_fsrs7(initial_stability: [f32; 4]) -> Result<[f32; 4]> {
    fill_initial_stabilities_fsrs7(&HashMap::from([
        (1, initial_stability[0]),
        (2, initial_stability[1]),
        (3, initial_stability[2]),
        (4, initial_stability[3]),
    ]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_smooth_initial_stabilities_fsrs7_is_monotonic() {
        let actual = smooth_initial_stabilities_fsrs7([4.0, 3.0, 2.0, 1.0]).unwrap();
        assert_eq!(actual, [4.0, 4.0, 4.0, 4.0]);
    }

    #[test]
    fn test_smooth_initial_stabilities_fsrs7_clamps_bounds() {
        let actual = smooth_initial_stabilities_fsrs7([0.0, 0.00005, 0.001, 200.0]).unwrap();
        assert_eq!(actual, [0.0001, 0.0001, 0.001, 100.0]);
    }
}
