use crate::{
    convertor::{convert_to_fsrs_items, RevlogEntry},
    model::ModelConfig,
    training::{train, TrainingConfig},
};
use burn::optim::AdamConfig;
use log::debug;
use pyo3::{types::*, *};

#[pyfunction]
fn py_train(py: Python, revlogs: &PyList) -> PyResult<()> {
    use burn_ndarray::NdArrayBackend;
    use burn_ndarray::NdArrayDevice;
    type Backend = NdArrayBackend<f32>;
    type AutodiffBackend = burn_autodiff::ADBackendDecorator<Backend>;

    let revlog: Vec<Vec<RevlogEntry>> = revlogs
        .into_iter()
        .map(|e| {
            let list: &PyList = pyo3::PyTryInto::try_into(e).unwrap();

            list.into_iter()
                .map(|e2| {
                    macro_rules! get {
                        ($key:expr) => {
                            e2.get_item($key)
                                .expect("Missing attr")
                                .extract()
                                .expect("Wrong type")
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
                .collect::<Vec<RevlogEntry>>() // https://regex101.com/r/sVxrTW/1
        })
        .collect();

    debug!("revlogs: {:?}", revlog);

    let items = convert_to_fsrs_items(revlog);

    let result = train::<AutodiffBackend>(
        "./tmp/fsrs",
        TrainingConfig::new(ModelConfig::new(), AdamConfig::new()),
        NdArrayDevice::Cpu,
        &items,
    );

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
