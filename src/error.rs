use snafu::Snafu;

#[derive(Snafu, Debug, PartialEq)]
pub enum FSRSError {
    NotEnoughData,
    Interrupted,
    InvalidParameters,
    OptimalNotFound,
    InvalidInput,
    InvalidDeckSize,
}

pub type Result<T, E = FSRSError> = std::result::Result<T, E>;
