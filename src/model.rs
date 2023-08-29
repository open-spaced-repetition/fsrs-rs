use burn::{
    config::Config,
    module::{Module, Param},
    tensor::{backend::Backend, Float, Tensor},
};

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    pub w: Param<Tensor<B, 1>>,
    pub freeze_stability: bool
}

impl<B: Backend<FloatElem = f32>> Model<B> {
    #[allow(clippy::new_without_default)]
    pub fn new(freeze_stability: bool) -> Self {
        Self {
            w: Param::from(Tensor::from_floats([
                0.4, 0.6, 2.4, 5.8, // initial stability
                4.93, 0.94, 0.86, 0.01, // difficulty
                1.49, 0.14, 0.94, // success
                2.18, 0.05, 0.34, 1.26, // failure
                0.29, 2.61, // hard penalty, easy bonus
            ])),
            freeze_stability
        }
    }

    fn w(&self) -> Tensor<B, 2> {
        self.w.val().unsqueeze().transpose()
    }

    pub fn power_forgetting_curve(&self, t: Tensor<B, 2>, s: Tensor<B, 2>) -> Tensor<B, 2> {
        (t / (s * 9) + 1).powf(-1.0)
    }

    fn stability_after_success(
        &self,
        last_s: Tensor<B, 2>,
        new_d: Tensor<B, 2>,
        r: Tensor<B, 2>,
        rating: Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        let batch_size = rating.dims()[0];
        let hard_penalty = Tensor::ones([batch_size, 1]).mask_where(
            rating.clone().equal_elem(2),
            self.w().slice([15..16]).unsqueeze(),
        );
        let easy_bonus = Tensor::ones([batch_size, 1])
            .mask_where(rating.equal_elem(4), self.w().slice([16..17]).unsqueeze());

        last_s.clone()
            * (self.w().slice([8..9]).reshape([1, 1]).exp()
                * (-new_d + 11)
                * (-self.w().slice([9..10]).reshape([1, 1]) * last_s.log()).exp()
                * (((-r + 1) * self.w().slice([10..11]).reshape([1, 1])).exp() - 1)
                * hard_penalty
                * easy_bonus
                + 1)
    }

    fn stability_after_failure(
        &self,
        last_s: Tensor<B, 2>,
        new_d: Tensor<B, 2>,
        r: Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        self.w().slice([11..12]).reshape([1, 1])
            * (-self.w().slice([12..13]).reshape([1, 1]) * new_d.log()).exp()
            * ((self.w().slice([13..14]).reshape([1, 1]) * (last_s + 1).log()).exp() - 1)
            * ((-r + 1) * self.w().slice([14..15]).reshape([1, 1])).exp()
    }

    fn mean_reversion(&self, new_d: Tensor<B, 2>) -> Tensor<B, 2> {
        self.w().slice([7..8]).reshape([1, 1])
            * (self.w().slice([4..5]).reshape([1, 1]) - new_d.clone())
            + new_d
    }

    fn init_stability(&self, rating: Tensor<B, 2>) -> Tensor<B, 2> {
        self.w()
            .squeeze::<1>(1)
            .select(0, rating.clone().squeeze::<1>(1).int() - 1)
            .unsqueeze()
            .transpose()
    }

    fn init_difficulty(&self, rating: Tensor<B, 2>) -> Tensor<B, 2> {
        self.w().slice([4..5]).reshape([1, 1])
            - self.w().slice([5..6]).reshape([1, 1]) * (rating - 3)
    }

    fn next_difficulty(&self, difficulty: Tensor<B, 2>, rating: Tensor<B, 2>) -> Tensor<B, 2> {
        difficulty - self.w().slice([6..7]).reshape([1, 1]) * (rating - 3)
    }

    fn step(
        &self,
        i: usize,
        delta_t: Tensor<B, 2>,
        rating: Tensor<B, 2>,
        stability: Tensor<B, 2>,
        difficulty: Tensor<B, 2>,
    ) -> (Tensor<B, 2>, Tensor<B, 2>) {
        if i == 0 {
            let new_s = self.init_stability(rating.clone());
            let new_d = self.init_difficulty(rating.clone());
            (new_s.clamp(0.1, 36500.0), new_d.clamp(1.0, 10.0))
        } else {
            let r = self.power_forgetting_curve(delta_t, stability.clone());
            let new_d = self.next_difficulty(difficulty.clone(), rating.clone());
            let new_d = self.mean_reversion(new_d);
            let new_d = new_d.clamp(1.0, 10.0);
            let s_recall = self.stability_after_success(
                stability.clone(),
                new_d.clone(),
                r.clone(),
                rating.clone(),
            );
            let s_forget: Tensor<B, 2> =
                self.stability_after_failure(stability.clone(), new_d.clone(), r.clone());
            let new_s = s_recall.mask_where(rating.clone().equal_elem(1), s_forget);
            // mask padding zeros for rating
            let new_s = new_s.mask_where(rating.equal_elem(0), stability);
            (new_s.clamp(0.1, 36500.0), new_d)
        }
    }

    pub fn forward(
        &self,
        delta_ts: Tensor<B, 2>,
        ratings: Tensor<B, 2, Float>,
    ) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let [seq_len, batch_size] = delta_ts.dims();
        let mut stability = Tensor::zeros([batch_size, 1]);
        let mut difficulty = Tensor::zeros([batch_size, 1]);
        for i in 0..seq_len {
            let delta_t = delta_ts.clone().slice([i..i + 1]).transpose();
            // [batch_size, 1]
            let rating = ratings.clone().slice([i..i + 1]).transpose();
            // [batch_size, 1]
            (stability, difficulty) = self.step(i, delta_t, rating, stability, difficulty);
            // dbg!("stability: {}", &stability);
            // dbg!("difficulty: {}", &difficulty);
            // dbg!()
        }
        (stability, difficulty)
    }
}

#[derive(Config, Debug)]
pub struct ModelConfig {
    #[config(default = false)]
    pub freeze_stability: bool
}

impl ModelConfig {
    pub fn init<B: Backend<FloatElem = f32>>(&self) -> Model<B> {
        Model::new(self.freeze_stability)
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::Data;
    use burn_autodiff::ADBackendDecorator;
    use burn_ndarray::NdArrayBackend;
    type Backend = ADBackendDecorator<NdArrayBackend<f32>>;
    use super::*;

    #[test]
    fn w() {
        use burn::tensor::Data;
        use burn_ndarray::NdArrayBackend;
        type Backend = NdArrayBackend<f32>;
        let model = Model::<Backend>::new(false);
        assert_eq!(
            model.w().to_data(),
            Data::from([
                [0.4],
                [0.6],
                [2.4],
                [5.8],
                [4.93],
                [0.94],
                [0.86],
                [0.01],
                [1.49],
                [0.14],
                [0.94],
                [2.18],
                [0.05],
                [0.34],
                [1.26],
                [0.29],
                [2.61]
            ])
        )
    }

    #[test]
    fn power_forgetting_curve() {
        use burn::tensor::Data;
        use burn_ndarray::NdArrayBackend;
        type Backend = NdArrayBackend<f32>;
        let model = Model::<Backend>::new();
        let delta_t = Tensor::<Backend, 2>::from_floats([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]]);
        let stability =
            Tensor::<Backend, 2>::from_floats([[1.0], [2.0], [3.0], [4.0], [4.0], [2.0]]);
        let retention = model.power_forgetting_curve(delta_t, stability);
        assert_eq!(
            retention.to_data(),
            Data::from([
                [1.0],
                [0.9473684],
                [0.9310345],
                [0.92307687],
                [0.9],
                [0.7826087]
            ])
        )
    }

    #[test]
    fn init_stability() {
        use burn::tensor::Data;
        use burn_ndarray::NdArrayBackend;
        type Backend = NdArrayBackend<f32>;
        let model = Model::<Backend>::new();
        let rating = Tensor::<Backend, 2>::from_floats([[1.0], [2.0], [3.0], [4.0], [1.0], [2.0]]);
        let stability = model.init_stability(rating);
        assert_eq!(
            stability.to_data(),
            Data::from([[0.4], [0.6], [2.4], [5.8], [0.4], [0.6]])
        )
    }

    #[test]
    fn init_difficulty() {
        use burn::tensor::Data;
        use burn_ndarray::NdArrayBackend;
        type Backend = NdArrayBackend<f32>;
        let model = Model::<Backend>::new();
        let rating = Tensor::<Backend, 2>::from_floats([[1.0], [2.0], [3.0], [4.0], [1.0], [2.0]]);
        let difficulty = model.init_difficulty(rating);
        assert_eq!(
            difficulty.to_data(),
            Data::from([[6.81], [5.87], [4.93], [3.9899998], [6.81], [5.87]])
        )
    }

    #[test]
    fn forward() {
        use burn_ndarray::NdArrayBackend;
        type Backend = NdArrayBackend<f32>;
        let model = Model::<Backend>::new();
        let delta_ts = Tensor::<Backend, 2>::from_floats([
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 2.0, 2.0],
        ]);
        let ratings = Tensor::<Backend, 2>::from_floats([
            [1.0, 2.0, 3.0, 4.0, 1.0, 2.0],
            [1.0, 2.0, 3.0, 4.0, 1.0, 2.0],
        ]);
        let (stability, difficulty) = model.forward(delta_ts, ratings);
        dbg!(&stability);
        dbg!(&difficulty);
    }

    #[test]
    fn next_difficulty() {
        let model = Model::<Backend>::new();
        let difficulty = Tensor::<Backend, 2>::from_floats([[5.0], [5.0], [5.0], [5.0]]);
        let rating = Tensor::<Backend, 2>::from_floats([[1.0], [2.0], [3.0], [4.0]]);
        let next_difficulty = model.next_difficulty(difficulty, rating);
        next_difficulty.clone().backward();
        assert_eq!(
            next_difficulty.to_data(),
            Data::from([[6.7200003], [5.86], [5.0], [4.14]])
        );
        let next_difficulty = model.mean_reversion(next_difficulty);
        next_difficulty.clone().backward();
        assert_eq!(
            next_difficulty.to_data(),
            Data::from([[6.7021003], [5.8507], [4.9993], [4.1478996]])
        )
    }

    #[test]
    fn next_stability() {
        let model = Model::<Backend>::new();
        let stability = Tensor::<Backend, 2>::from_floats([[5.0], [5.0], [5.0], [5.0]]);
        let difficulty = Tensor::<Backend, 2>::from_floats([[1.0], [2.0], [3.0], [4.0]]);
        let retention = Tensor::<Backend, 2>::from_floats([[0.9], [0.8], [0.7], [0.6]]);
        let rating = Tensor::<Backend, 2>::from_floats([[1.0], [2.0], [3.0], [4.0]]);
        let s_recall = model.stability_after_success(
            stability.clone(),
            difficulty.clone(),
            retention.clone(),
            rating.clone(),
        );
        s_recall.clone().backward();
        assert_eq!(
            s_recall.to_data(),
            Data::from([[22.454704], [14.560361], [51.15574], [152.6869]])
        );
        let s_forget = model.stability_after_failure(stability, difficulty, retention);
        s_forget.clone().backward();
        assert_eq!(
            s_forget.to_data(),
            Data::from([[2.074517], [2.2729328], [2.526406], [2.8247323]])
        );
        let next_stability = s_recall.mask_where(rating.clone().equal_elem(1), s_forget);
        next_stability.clone().backward();
        assert_eq!(
            next_stability.to_data(),
            Data::from([[2.074517], [14.560361], [51.15574], [152.6869]])
        )
    }
}
