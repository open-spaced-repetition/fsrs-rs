use snafu::Snafu;

#[derive(Snafu, Debug)]
pub enum FsrsError {
    NotEnoughData,
}

pub type Result<T, E = FsrsError> = std::result::Result<T, E>;
