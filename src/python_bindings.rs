use burn::optim::AdamConfig;
use pyo3::{*, types::*};
use crate::{training::{TrainingConfig, train}, model::ModelConfig};

#[pyfunction]
fn py_train() -> PyResult<PyList> {
    use burn_ndarray::NdArrayDevice;
    use burn_ndarray::NdArrayBackend;
    type Backend = NdArrayBackend<f32>;
    type AutodiffBackend = burn_autodiff::ADBackendDecorator<Backend>;

    let result = train::<AutodiffBackend>(
        "./tmp/fsrs",
        TrainingConfig::new(ModelConfig::new(), AdamConfig::new()),
        NdArrayDevice::Cpu
    );
    Ok(PyList::from(result))
}

#[pymodule]
fn fsrs_optimizer_rust(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_train, m)?)?;
    Ok(())
}