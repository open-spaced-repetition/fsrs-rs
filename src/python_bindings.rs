use crate::{
    convertor::{convert_to_fsrs_items, RevlogEntry},
    model::ModelConfig,
    training::{train, TrainingConfig},
};
use burn::optim::AdamConfig;
use log::debug;
use pyo3::{types::*, *};
use std::format;

#[pyfunction]
fn py_train(_py: Python, revlogs: &PyList) -> PyResult<()> {
    use burn_ndarray::NdArrayBackend;
    use burn_ndarray::NdArrayDevice;
    type Backend = NdArrayBackend<f32>;
    type AutodiffBackend = burn_autodiff::ADBackendDecorator<Backend>;

    let revlog: Vec<RevlogEntry> = revlogs
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
                button_chosen: get!("button_chosen"),
                interval: get!("interval"),
                last_interval: get!("last_interval"),
                ease_factor: get!("ease_factor"),
                taken_millis: get!("taken_millis"),
                review_kind: get!("review_kind"),
                delta_t: get!("delta_t"),
                i: get!("i"),
                r_history: get!("r_history"),
                t_history: get!("t_history"),
            }

        })
        .collect();

    debug!("revlogs: {:?}", revlog);

    let items = convert_to_fsrs_items(vec![revlog]);

    let result = train::<AutodiffBackend>(
        "./tmp/fsrs",
        TrainingConfig::new(ModelConfig::new(), AdamConfig::new()),
        NdArrayDevice::Cpu,
        &items,
    );

    println!("{:?}", result);
    //let list = PyList::empty(py);
    //list.append(item)
    //let list = PyList::new(py, result);
    Ok(())
}

#[pymodule]
fn fsrs_optimizer_rust(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_train, m)?)?;
    Ok(())
}
