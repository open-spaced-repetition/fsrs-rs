use burn::{
    module::{Module,
    Param,},
    nn::{loss::CrossEntropyLoss},
    tensor::{
        backend::{ADBackend, Backend},
        Tensor,
    },
    train::{ClassificationOutput, TrainOutput, TrainStep, ValidStep},
};

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    w0: Param<Tensor<B, 1>>,
    w1: Param<Tensor<B, 1>>,
    w2: Param<Tensor<B, 1>>,
    w3: Param<Tensor<B, 1>>,
    w4: Param<Tensor<B, 1>>,
    w5: Param<Tensor<B, 1>>,
    w6: Param<Tensor<B, 1>>,
    w7: Param<Tensor<B, 1>>,
    w8: Param<Tensor<B, 1>>,
    w9: Param<Tensor<B, 1>>,
    w10: Param<Tensor<B, 1>>,
    w11: Param<Tensor<B, 1>>,
    w12: Param<Tensor<B, 1>>,
    w13: Param<Tensor<B, 1>>,
    w14: Param<Tensor<B, 1>>,
    w15: Param<Tensor<B, 1>>,
    w16: Param<Tensor<B, 1>>,
}

impl<B: Backend> Model<B> {
    pub fn new() -> Self {
        // [0.4, 0.6, 2.4, 5.8, 4.93, 0.94, 0.86, 0.01, 1.49, 0.14, 0.94, 2.18, 0.05, 0.34, 1.26, 0.29, 2.61]
        let w0 = Param::from(Tensor::from_floats([0.4]));
        let w1 = Param::from(Tensor::from_floats([0.6]));
        let w2 = Param::from(Tensor::from_floats([2.4]));
        let w3 = Param::from(Tensor::from_floats([5.8]));
        let w4 = Param::from(Tensor::from_floats([4.93]));
        let w5 = Param::from(Tensor::from_floats([0.94]));
        let w6 = Param::from(Tensor::from_floats([0.86]));
        let w7 = Param::from(Tensor::from_floats([0.01]));
        let w8 = Param::from(Tensor::from_floats([1.49]));
        let w9 = Param::from(Tensor::from_floats([0.14]));
        let w10 = Param::from(Tensor::from_floats([0.94]));
        let w11 = Param::from(Tensor::from_floats([2.18]));
        let w12 = Param::from(Tensor::from_floats([0.05]));
        let w13 = Param::from(Tensor::from_floats([0.34]));
        let w14 = Param::from(Tensor::from_floats([1.26]));
        let w15 = Param::from(Tensor::from_floats([0.29]));
        let w16 = Param::from(Tensor::from_floats([2.61]));
        Self {
            w0,
            w1,
            w2,
            w3,
            w4,
            w5,
            w6,
            w7,
            w8,
            w9,
            w10,
            w11,
            w12,
            w13,
            w14,
            w15,
            w16,
        }
    }

    fn stability_after_success(&self, last_s: Tensor<B, 1>, new_d: Tensor<B, 1>, r: Tensor<B, 1>, rating: Tensor<B, 1>) -> Tensor<B, 1> {
        let hard_penalty = Tensor::ones_like(&rating).mask_where(rating.equal_elem(2), self.w15.val());
        let easy_bonus = Tensor::ones_like(&rating).mask_where(rating.equal_elem(4), self.w16.val());
        let new_s = last_s * (self.w8.val().exp() *
            (-new_d + 11) *
            (-self.w9.val() * last_s.log()).exp() *
            ((r * -self.w10.val()).exp() - 1) * 
            hard_penalty * 
            easy_bonus + 1);
        new_s
    }

    fn stability_after_failure(&self, last_s: Tensor<B, 1>, new_d: Tensor<B, 1>, r: Tensor<B, 1>) -> Tensor<B, 1> {
        let new_s = self.w11.val() *
            (-self.w12.val() * new_d.log()).exp() *
            ((self.w13.val() * (last_s + 1).log()).exp() - 1) *
            ((-r + 1) * self.w14.val()).exp();
        new_s
    }

    fn step(&self, i: i32, input: Tensor<B, 2>, state: Tensor<B, 2>) -> Tensor<B, 2> {
        if i == 0 {
            let keys = Tensor::from_ints([1, 2, 3, 4]).repeat(1, input.shape()[0]);
            let index = 
        } else {
            
        }
    }


    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 2> {
        let [batch_size, heigth, width] = input.dims();

        let x = input.reshape([batch_size, 1, heigth, width]).detach();
        let x = self.conv1.forward(x);
        let x = self.conv2.forward(x);
        let x = self.conv3.forward(x);

        let [batch_size, channels, heigth, width] = x.dims();
        let x = x.reshape([batch_size, channels * heigth * width]);

        let x = self.dropout.forward(x);
        let x = self.fc1.forward(x);
        let x = self.activation.forward(x);

        self.fc2.forward(x)
    }

    pub fn forward_classification(&self, item: MNISTBatch<B>) -> ClassificationOutput<B> {
        let targets = item.targets;
        let output = self.forward(item.images);
        let loss = CrossEntropyLoss::new(None);
        let loss = loss.forward(output.clone(), targets.clone());

        ClassificationOutput {
            loss,
            output,
            targets,
        }
    }
}


impl<B: ADBackend> TrainStep<MNISTBatch<B>, ClassificationOutput<B>> for Model<B> {
    fn step(&self, item: MNISTBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_classification(item);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<MNISTBatch<B>, ClassificationOutput<B>> for Model<B> {
    fn step(&self, item: MNISTBatch<B>) -> ClassificationOutput<B> {
        self.forward_classification(item)
    }
}
