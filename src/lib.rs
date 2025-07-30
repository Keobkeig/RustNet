extern crate rand;
use rand::Rng;
use std::ops::{Add, Mul, Sub};

/// A simple Matrix struct with rows, columns, and a flat vector for data.
#[derive(Debug, Clone)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

impl Matrix {
    /// Creates a new matrix with the given dimensions, filled with zeros.
    pub fn new(rows: usize, cols: usize) -> Self {
        Matrix {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    /// Creates a column matrix (n×1) from a given vector.
    pub fn from_vec(vec: Vec<f64>) -> Self {
        let rows = vec.len();
        Matrix {
            rows,
            cols: 1,
            data: vec,
        }
    }

    /// Creates a matrix of the given dimensions with random values in [0, 1).
    pub fn random(rows: usize, cols: usize) -> Matrix {
        let mut rng = rand::rng();
        let mut data = Vec::with_capacity(rows * cols);
        for _ in 0..(rows * cols) {
            data.push(rng.random_range(0.0..1.0));
        }
        Matrix { rows, cols, data }
    }

    /// Returns a new matrix by applying a function elementwise.
    pub fn elementwise_apply<F>(&self, func: F) -> Matrix
    where
        F: Fn(f64) -> f64,
    {
        let data = self.data.iter().map(|&x| func(x)).collect();
        Matrix {
            rows: self.rows,
            cols: self.cols,
            data,
        }
    }

    /// Returns the Hadamard (elementwise) product of two matrices.
    pub fn elementwise_multiply(&self, other: &Matrix) -> Matrix {
        if self.rows != other.rows || self.cols != other.cols {
            panic!("Dimension mismatch for elementwise multiplication");
        }
        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a * b)
            .collect();
        Matrix {
            rows: self.rows,
            cols: self.cols,
            data,
        }
    }

    /// Returns a new matrix scaled by the given scalar.
    pub fn scalar_mul(&self, scalar: f64) -> Matrix {
        let data = self.data.iter().map(|&x| x * scalar).collect();
        Matrix {
            rows: self.rows,
            cols: self.cols,
            data,
        }
    }

    /// Returns the transpose of the matrix.
    pub fn transpose(&self) -> Matrix {
        let mut result = vec![0.0; self.rows * self.cols];
        for i in 0..self.rows {
            for j in 0..self.cols {
                result[j * self.rows + i] = self.data[i * self.cols + j];
            }
        }
        Matrix {
            rows: self.cols,
            cols: self.rows,
            data: result,
        }
    }
}

impl Add for Matrix {
    type Output = Matrix;

    fn add(self, other: Matrix) -> Matrix {
        &self + &other
    }
}

impl Add<&Matrix> for &Matrix {
    type Output = Matrix;

    fn add(self, other: &Matrix) -> Matrix {
        if self.rows != other.rows || self.cols != other.cols {
            panic!("Matrix dimensions must match for addition");
        }
        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a + b)
            .collect();
        Matrix {
            rows: self.rows,
            cols: self.cols,
            data,
        }
    }
}

impl Add<Matrix> for &Matrix {
    type Output = Matrix;

    fn add(self, other: Matrix) -> Matrix {
        self + &other
    }
}

impl Add<&Matrix> for Matrix {
    type Output = Matrix;

    fn add(self, other: &Matrix) -> Matrix {
        &self + other
    }
}

impl Sub for Matrix {
    type Output = Matrix;

    fn sub(self, other: Matrix) -> Matrix {
        &self - &other
    }
}

impl Sub<&Matrix> for &Matrix {
    type Output = Matrix;

    fn sub(self, other: &Matrix) -> Matrix {
        if self.rows != other.rows || self.cols != other.cols {
            panic!("Matrix dimensions must match for subtraction");
        }
        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a - b)
            .collect();
        Matrix {
            rows: self.rows,
            cols: self.cols,
            data,
        }
    }
}

impl Sub<Matrix> for &Matrix {
    type Output = Matrix;

    fn sub(self, other: Matrix) -> Matrix {
        self - &other
    }
}

impl Sub<&Matrix> for Matrix {
    type Output = Matrix;

    fn sub(self, other: &Matrix) -> Matrix {
        &self - other
    }
}

//dot prod
impl Mul for Matrix {
    type Output = Matrix;

    fn mul(self, other: Matrix) -> Matrix {
        &self * &other
    }
}

impl Mul<&Matrix> for &Matrix {
    type Output = Matrix;

    fn mul(self, other: &Matrix) -> Matrix {
        if self.cols != other.rows {
            panic!("Matrix dimensions must agree for multiplication");
        }
        let mut result = vec![0.0; self.rows * other.cols];
        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = 0.0;
                for k in 0..self.cols {
                    sum += self.data[i * self.cols + k] * other.data[k * other.cols + j];
                }
                result[i * other.cols + j] = sum;
            }
        }
        Matrix {
            rows: self.rows,
            cols: other.cols,
            data: result,
        }
    }
}

impl Mul<Matrix> for &Matrix {
    type Output = Matrix;

    fn mul(self, other: Matrix) -> Matrix {
        self * &other
    }
}

impl Mul<&Matrix> for Matrix {
    type Output = Matrix;

    fn mul(self, other: &Matrix) -> Matrix {
        &self * other
    }
}

pub struct Network {
    layers: Vec<usize>,
    weights: Vec<Matrix>,
    biases: Vec<Matrix>,
    activations: Vec<Matrix>,
    learning_rate: f64,
}

impl Network {
    /// Creates a new network with the specified layer sizes and learning rate.
    /// For example, `Network::new(vec![2, 3, 1], 0.5)` creates a network with 2 inputs,
    /// one hidden layer with 3 neurons, and 1 output neuron.
    pub fn new(layers: Vec<usize>, learning_rate: f64) -> Self {
        let mut weights = Vec::new();
        let mut biases = Vec::new();

        for i in 0..layers.len() - 1 {
            // Weight matrix shape: (next_layer_size, current_layer_size)
            weights.push(Matrix::random(layers[i + 1], layers[i]));
            // Bias vector shape: (next_layer_size, 1)
            biases.push(Matrix::random(layers[i + 1], 1));
        }

        Network {
            layers,
            weights,
            biases,
            activations: Vec::new(),
            learning_rate,
        }
    }

    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    fn sigmoid_derivative(a: f64) -> f64 {
        a * (1.0 - a)
    }

    pub fn feedforward(&mut self, input: Matrix) -> Matrix {
        if input.rows != self.layers[0] || input.cols != 1 {
            panic!("Input dimensions do not match network input layer");
        }

        self.activations.clear();
        self.activations.push(input.clone());

        let mut current = input;
        for i in 0..self.weights.len() {
            // z = W * a + b (W: weights, a: activations, b: biases)
            let z = &self.weights[i] * &current + &self.biases[i];
            // a = f(z) (f: activation function)
            let activated = z.elementwise_apply(|x| Network::sigmoid(x));
            self.activations.push(activated.clone());
            current = activated;
        }
        current
    }

    pub fn backpropagate(&mut self, target: Matrix) {
        let num_layers = self.activations.len();
        // output activation (a^L)
        let output = &self.activations[num_layers - 1];
        let error = target - output;
        // delta = error ∘ f'(output) (elementwise multiplication)
        let output_derivative = output.elementwise_apply(|a| Network::sigmoid_derivative(a));
        let mut delta = error.elementwise_multiply(&output_derivative);

        //update W and b for output
        // The weight connecting the last hidden layer to the output layer is at index num_layers - 2.
        let prev_activation = &self.activations[num_layers - 2];
        let prev_activation_t = prev_activation.transpose();
        let weight_grad = &delta * prev_activation_t;
        self.weights[num_layers - 2] =
            &self.weights[num_layers - 2] + weight_grad.scalar_mul(self.learning_rate);
        self.biases[num_layers - 2] =
            &self.biases[num_layers - 2] + delta.scalar_mul(self.learning_rate);

        for l in (1..(num_layers - 1)).rev() {
            // the weights connecting layer l to l+1 are at index l.
            let weight_next = &self.weights[l];
            // error for hidden = W^(l)^T * delta
            let error_hidden = weight_next.transpose() * &delta;
            // delta for hidden = error_hidden ∘ f'(activation[l])
            let hidden_activation = &self.activations[l];
            let hidden_derivative =
                hidden_activation.elementwise_apply(|a| Network::sigmoid_derivative(a));
            delta = error_hidden.elementwise_multiply(&hidden_derivative);

            // update layer l-1
            let prev_activation = &self.activations[l - 1];
            let prev_activation_t = prev_activation.transpose();
            let weight_grad = &delta * prev_activation_t;
            self.weights[l - 1] = &self.weights[l - 1] + weight_grad.scalar_mul(self.learning_rate);
            self.biases[l - 1] = &self.biases[l - 1] + delta.scalar_mul(self.learning_rate);
        }
    }

    pub fn train(&mut self, inputs: Vec<Vec<f64>>, targets: Vec<Vec<f64>>, epochs: usize) {
        if inputs.len() != targets.len() {
            panic!("Number of inputs and targets must match");
        }
        for epoch in 0..epochs {
            // if epochs < 100 || epoch % (epochs / 100) == 0 {
            //     println!("Epoch: {}", epoch);
            // }
            for (inp, tar) in inputs.iter().zip(targets.iter()) {
                let input_matrix = Matrix::from_vec(inp.to_vec());
                let target_matrix = Matrix::from_vec(tar.to_vec());
                let _ = self.feedforward(input_matrix);
                self.backpropagate(target_matrix);
            }
        }
    }
}
