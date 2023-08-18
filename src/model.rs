use burn::{
    module::{Param, Module},
    tensor::{backend::Backend, Float, Tensor}, config::Config,
};


#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    w: Param<Tensor<B, 1>>,
}

impl<B: Backend> Model<B> {
    pub fn new() -> Self {
        Self {
            w: Param::from(Tensor::from_floats([
                0.4, 0.6, 2.4, 5.8, // initial stability
                4.93, 0.94, 0.86, 0.01, // difficulty
                1.49, 0.14, 0.94, // success
                2.18, 0.05, 0.34, 1.26, // failure
                0.29, 2.61, // hard penalty, easy bonus
            ])),
        }
    }

    pub fn power_forgetting_curve(&self, t: Tensor<B, 1>, s: Tensor<B, 1>) -> Tensor<B, 1> {
        let retrievability = (t / (s * 9) + 1).powf(-1.0);
        retrievability
    }

    fn stability_after_success(
        &self,
        last_s: Tensor<B, 1>,
        new_d: Tensor<B, 1>,
        r: Tensor<B, 1>,
        rating: Tensor<B, 1>,
    ) -> Tensor<B, 1> {
        let batch_size = rating.dims()[0];
        let hard_penalty = Tensor::ones([batch_size])
            .mask_where(rating.clone().equal_elem(2), self.w.val().slice([15..16]));
        let easy_bonus = Tensor::ones([batch_size])
            .mask_where(rating.clone().equal_elem(4), self.w.val().slice([16..17]));
        let new_s = last_s.clone()
            * (self.w.val().slice([8..9]).exp()
                * (-new_d + 11)
                * (-self.w.val().slice([9..10]) * last_s.log()).exp()
                * (((-r + 1) * self.w.val().slice([10..11])).exp() - 1)
                * hard_penalty
                * easy_bonus
                + 1);
        new_s
    }

    fn stability_after_failure(
        &self,
        last_s: Tensor<B, 1>,
        new_d: Tensor<B, 1>,
        r: Tensor<B, 1>,
    ) -> Tensor<B, 1> {
        let new_s = self.w.val().slice([11..12])
            * (-self.w.val().slice([12..13]) * new_d.log()).exp()
            * ((self.w.val().slice([13..14]) * (last_s + 1).log()).exp() - 1)
            * ((-r + 1) * self.w.val().slice([14..15])).exp();
        new_s
    }

    fn step(
        &self,
        i: usize,
        delta_t: Tensor<B, 1>,
        rating: Tensor<B, 1>,
        stability: Tensor<B, 1>,
        difficulty: Tensor<B, 1>,
    ) -> (Tensor<B, 1>, Tensor<B, 1>) {
        if i == 0 {
            let new_s = self.w.val().select(0, rating.clone().int() - 1);
            let new_d = self.w.val().slice([4..5]) - self.w.val().slice([5..6]) * (rating - 3);
            (new_s, new_d)
        } else {
            let r = self.power_forgetting_curve(delta_t, stability.clone());
            dbg!(&r);
            let new_d = difficulty.clone() - self.w.val().slice([6..7]) * (rating.clone() - 3);
            // let new_d = new_d.clamp(1.0, 10.0);
            // TODO: consider constraining the associated type `<B as Backend>::FloatElem` to `{float}` or calling a method that returns `<B as Backend>::FloatElem`
            dbg!(&new_d);
            let s_recall = self.stability_after_success(
                stability.clone(),
                new_d.clone(),
                r.clone(),
                rating.clone(),
            );
            let s_forget = self.stability_after_failure(stability, new_d.clone(), r);
            let new_s = s_recall.mask_where(rating.equal_elem(1), s_forget);
            (new_s, new_d)
        }
    }

    pub fn forward(
        &self,
        delta_ts: Tensor<B, 2>,
        ratings: Tensor<B, 2, Float>,
    ) -> (Tensor<B, 1>, Tensor<B, 1>) {
        let [seq_len, batch_size] = delta_ts.dims();
        let mut stability = Tensor::zeros([batch_size]);
        let mut difficulty = Tensor::zeros([batch_size]);
        for i in 0..seq_len {
            let delta_t = delta_ts.clone().slice([i..i + 1]).squeeze(0);
            let rating = ratings.clone().slice([i..i + 1]).squeeze(0);
            dbg!(&delta_t);
            dbg!(&rating);
            (stability, difficulty) = self.step(i, delta_t, rating, stability, difficulty);
            dbg!(&stability);
            dbg!(&difficulty);
            dbg!()
        }
        (stability, difficulty)
    }
}


#[derive(Config, Debug)]
pub struct ModelConfig {
}

impl ModelConfig {
    pub fn init<B: Backend>(&self) -> Model<B> {
        Model::new()
    }
}

#[test]
fn test() {
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
    println!("stability {:?}", stability);
    println!("difficulty {:?}", difficulty);
}
