use snafu::Snafu;

/// Represents an error that can occur during FSRS operations.
#[derive(Snafu, Debug, PartialEq)]
pub enum FSRSError {
    NotEnoughData,
    Interrupted,
    InvalidParameters,
    OptimalNotFound,
    InvalidInput,
    InvalidDeckSize,
}

/// A type alias for [`Result`] with [`FSRSError`] as the error type.
pub type Result<T, E = FSRSError> = std::result::Result<T, E>;
