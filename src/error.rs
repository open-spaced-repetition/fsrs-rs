use snafu::Snafu;

#[derive(Snafu, Debug)]
pub enum FSRSError {
    NotEnoughData,
    Interrupted,
    InvalidWeights,
    OptimalNotFound,
    InvalidInput,
}

pub type Result<T, E = FSRSError> = std::result::Result<T, E>;
