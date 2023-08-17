use burn::{
    module::{Module,
    Param,},
    tensor::{
        backend::Backend,
        Tensor, Float
    },
};

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    w: Param<Tensor<B, 1>>,
}

impl<B: Backend> Model<B> {
    pub fn new() -> Self {
        // [0.4, 0.6, 2.4, 5.8, 4.93, 0.94, 0.86, 0.01, 1.49, 0.14, 0.94, 2.18, 0.05, 0.34, 1.26, 0.29, 2.61]
        let w: Param<Tensor<B, 1>> = Param::from(Tensor::from_floats([0.4, 0.6, 2.4, 5.8, 4.93, 0.94, 0.86, 0.01, 1.49, 0.14, 0.94, 2.18, 0.05, 0.34, 1.26, 0.29, 2.61]));
        Self {
            w
        }
    }

    fn power_forgetting_curve(&self, t: Tensor<B, 1>, s: Tensor<B, 1>) -> Tensor<B, 1> {
        let retrievability = (Tensor::ones_like(&t) + t / (self.w.slice([0..1]) * s * 9)).powf(-1.0);
        retrievability
    }

    fn stability_after_success(&self, last_s: Tensor<B, 1>, new_d: Tensor<B, 1>, r: Tensor<B, 1>, rating: Tensor<B, 1>) -> Tensor<B, 1> {
        let hard_penalty = Tensor::ones_like(&rating).mask_where(rating.equal_elem(2), self.w.slice([15..16]));
        let easy_bonus = Tensor::ones_like(&rating).mask_where(rating.equal_elem(4), self.w.slice([16..17]));
        let new_s = last_s * (self.w.slice([8..9]).exp() *
            (-new_d + 11) *
            (-self.w.slice([9..10]) * last_s.log()).exp() *
            ((r * -self.w.slice([10..11])).exp() - 1) * 
            hard_penalty * 
            easy_bonus + 1);
        new_s
    }

    fn stability_after_failure(&self, last_s: Tensor<B, 1>, new_d: Tensor<B, 1>, r: Tensor<B, 1>) -> Tensor<B, 1> {
        let new_s = self.w.slice([11..12]) *
            (-self.w.slice([12..13]) * new_d.log()).exp() *
            ((self.w.slice([13..14]) * (last_s + 1).log()).exp() - 1) *
            ((-r + 1) * self.w.slice([14..15])).exp();
        new_s
    }

    fn step(&self, i: i32, delta_t: Tensor<B, 1>, rating: Tensor<B, 1>, stability: Tensor<B, 1>, difficulty: Tensor<B, 1>) -> (Tensor<B, 1>, Tensor<B, 1>) {
        if i == 0 {
            let new_s = self.w.select(0, rating.int()); // TODO: int
            let new_d = self.w.slice([4..5]) - self.w.slice([5..6]) * rating;
            (new_s, new_d)
        } else {
            let r = self.power_forgetting_curve(delta_t, stability);
            let new_d = difficulty + self.w.slice([1..2]) * (rating - self.w.slice([2..3]) * difficulty);
            let new_d = new_d.clip(1, 10); // TODO: clip
            let new_s = self.stability_after_success(stability, new_d, r, rating).mask_where(rating.equal_elem(1), self.stability_after_failure(stability, new_d, r));
            (new_s, new_d)
        }
    }

    pub fn forward(&self, delta_ts: Tensor<B, 2>, ratings: Tensor<B, 2, Float>, stabilitys: Tensor<B, 2>, difficultys: Tensor<B, 2>) -> (Tensor<B, 1>, Tensor<B, 1>) {
        let [seq_len, batch_size] = delta_ts.dims();
        let mut stability = Tensor::zeros([batch_size]);
        let mut difficulty = Tensor::zeros([batch_size]);
        for (i, delta_t, rating, stability, diffculty) in ().enumerate() { // TODO: enumerate
            let (stability, difficulty) = self.step(i, delta_t, rating, stability, difficulty);
        }
        (stability, difficulty)
    }
}