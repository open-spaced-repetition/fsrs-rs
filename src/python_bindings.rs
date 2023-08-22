use crate::{
    convertor::{anki_to_fsrs, RevlogEntry},
    model::ModelConfig,
    training::{train, TrainingConfig},
};
use burn::optim::AdamConfig;
use log::debug;
use pyo3::{types::*, *};
use std::format;

#[pyfunction(name = "train")]
fn py_train(revlogs: &PyList) -> PyResult<Vec<f32>> {
    use burn_ndarray::NdArrayBackend;
    use burn_ndarray::NdArrayDevice;
    type Backend = NdArrayBackend<f32>;
    type AutodiffBackend = burn_autodiff::ADBackendDecorator<Backend>;

    let revlogs: Vec<RevlogEntry> = revlogs
        .into_iter()
        .map(|e| {
            let obj: &PyDict = pyo3::PyTryInto::try_into(e).unwrap();

            macro_rules! get {
                ($key:expr) => {
                    obj.get_item($key)
                        .expect(&format!("Missing attr: {}", $key))
                        .extract()
                        .expect(&format!("Wrong type for key: {}", $key))
                };
            }

            RevlogEntry {
                id: get!("id"),
                cid: get!("cid"),
                usn: get!("usn"),
                button_chosen: get!("ease"),
                interval: get!("ivl"),
                last_interval: get!("lastIvl"),
                ease_factor: get!("factor"),
                taken_millis: get!("time"),
                review_kind: get!("type"),
                delta_t: 0,
                i: 0,
                r_history: vec![],
                t_history: vec![],
            }
        })
        .collect();

    debug!("revlogs: {:?}", revlogs);

    let items = anki_to_fsrs(revlogs);

    let result = train::<AutodiffBackend>(
        "./tmp/fsrs",
        TrainingConfig::new(ModelConfig::new(), AdamConfig::new()),
        NdArrayDevice::Cpu,
        &items,
    );

    println!("{:?}", result);

    Ok(result)
}

#[pymodule]
fn fsrs_optimizer_rust(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_train, m)?)?;
    Ok(())
}
