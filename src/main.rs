extern crate rand;
use rand::Rng;

//add the override for the + operator
use std::ops::{Add, Mul, Sub};

#[derive(Debug, Clone)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>, 
}



impl Add for Matrix {
    type Output = Matrix;

    fn add(self, other: Matrix) -> Matrix {
        if self.rows != other.rows || self.cols != other.cols {
            panic!("Matrix dimensions must match");
        }

        let buffer: Vec<f64> = self.data.iter().zip(other.data.iter()).map(|(a, b)| a + b).collect();

        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: buffer,
        }
    }
}

impl Sub for Matrix {
    type Output = Matrix;

    fn sub(self, other: Matrix) -> Matrix {
        if self.rows != other.rows || self.cols != other.cols {
            panic!("Matrix dimensions must match");
        }

        let buffer: Vec<f64> = self.data.iter().zip(other.data.iter()).map(|(a, b)| a - b).collect();

        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: buffer,
        }
    }
}

//override the * operator to be dot product
impl Mul for Matrix {
    type Output = Matrix;

    fn mul(self, other: Matrix) -> Matrix {
        if self.cols != other.rows {
            panic!("Matrix dimensions must match");
        }

        let mut buffer = vec![0.0; self.rows * other.cols];

        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = 0.0;
                for k in 0..self.cols {
                    sum += self.data[i * self.cols + k] * other.data[k * other.cols + j];
                }
                buffer[i * other.cols + j] = sum;
            }
        }

        Matrix {
            rows: self.rows,
            cols: other.cols,
            data: buffer,
        }
    }
}

impl Matrix {
    pub fn new(rows: usize, cols: usize) -> Self {
        Matrix {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    pub fn from(vec: Vec<f64>) -> Self {
        let rows = vec.len();
        Matrix {
            rows,
            cols: 1,
            data: vec,
        }
    }

    pub fn random(rows: usize, cols: usize) -> Matrix {
        let mut buffer = Vec::<f64>::with_capacity(rows * cols);
        
        for _ in 0..rows * cols {
            buffer.push(rand::rng().random_range(0.0..1.0));
        }

        Matrix {
            rows,
            cols,
            data: buffer,
        }
    }

    pub fn copy(&self) -> Matrix {
        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: self.data.clone(),
        }
    }

    pub fn elementwise_multiply(self, errors: &Matrix) -> Matrix {
        if self.data.len() != errors.data.len() {
            panic!("Cannot perform Hadamard product with non-identically shaped matrices");
        }

        let buffer: Vec<f64> = self.data.iter().zip(errors.data.iter()).map(|(a, b)| a * b).collect();

        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: buffer,
        }
    }

    pub fn transpose(&self) -> Matrix {
        let mut transposed_data = vec![0.0; self.cols * self.rows];
        for i in 0..self.rows {
            for j in 0..self.cols {
                transposed_data[j * self.rows + i] = self.data[i * self.cols + j];
            }
        }
        Matrix {
            rows: self.cols,
            cols: self.rows,
            data: transposed_data,
        }
    }
}

impl Iterator for Matrix {
    type Item = f64;

    fn next(&mut self) -> Option<f64> {
        self.data.pop()
    }
}
pub struct Network {
    layers: Vec<usize>,
    weights: Vec<Matrix>,
    biases: Vec<Matrix>,
    data: Vec<Matrix>,
    activation: fn(f64) -> f64,
    activation_derivative: fn(f64) -> f64,
    learning_rate: f64,
}

impl Network {
    pub fn new(layers: Vec<usize>, activation: fn(f64) -> f64, activation_derivative: fn(f64) -> f64, learning_rate: f64) -> Self {
        let mut weights = vec![];
        let mut biases = vec![];

        for i in 0..layers.len() - 1 {
            weights.push(Matrix::random(layers[i + 1], layers[i]));
            biases.push(Matrix::random(layers[i + 1], 1));
        }

        Network {
            layers,
            weights,
            biases,
            data: vec![],
            activation,
            activation_derivative,
            learning_rate,
        }
    }

    pub fn feedforward(&mut self, inputs: Matrix) -> Matrix {
        assert!(self.layers[0] == inputs.data.len(), "Input data must match the input layer size");

        let mut current = inputs; 
        self.data = vec![current.clone()];

        for i in 0..self.layers.len() - 1 {
            current = (self.weights[i].clone() * current.clone()) + self.biases[i].clone();
            current.data = current.data.iter().map(|x| (self.activation)(*x)).collect();
            self.data.push(current.clone());
        }

        current
    }

    pub fn back_propogate(&mut self, inputs: &Matrix, targets: Matrix) {
        let output = self.data.last().unwrap();
        let mut errors = targets - output.clone();
        for i in (0..self.layers.len() - 1).rev() {
                // Compute layer-specific gradients
            let layer_output = &self.data[i];
            
            // Compute derivatives for this layer
            let derivatives: Vec<f64> = layer_output.data.iter()
                .map(|&x| (self.activation_derivative)(x))
                .collect();

            // Scale gradients by error and learning rate
            let gradients: Vec<f64> = derivatives.iter()
                .zip(errors.data.iter())
                .map(|(&deriv, &err)| deriv * err * self.learning_rate)
                .collect();

            let gradient_matrix = Matrix { 
                rows: gradients.len(),
                cols: 1,
                data: gradients.clone(),
            };

            let deltas = gradient_matrix.clone() * layer_output.transpose();
            self.weights[i] = self.weights[i].clone() + deltas;
            self.biases[i] = self.biases[i].clone() + gradient_matrix.clone();

            if i > 0 {
                errors = self.weights[i].transpose() * errors;
            }
        }
    }

    pub fn train(&mut self, inputs: Vec<Vec<f64>>, targets: Vec<Vec<f64>>, epochs: usize) {
        for i in 0..epochs {
            if epochs < 100 || i % (epochs / 100) == 0 {
                println!("Epoch: {}", i);
            }
            for j in 0..inputs.len() {
                let outputs = self.feedforward(Matrix::from(inputs[j].clone()));
                self.back_propogate(&outputs, Matrix::from(targets[j].clone()));
            }
        }
        
    }
}

fn activation(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp()) // Sigmoid
}   

fn activation_derivative(x: f64) -> f64 {
    let fx = activation(x);
    fx * (1.0 - fx) // Sigmoid derivative
}


pub fn main() {
    let inputs: Vec<Vec<f64>> = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];

    let targets: Vec<Vec<f64>> = vec![
        vec![0.0],
        vec![1.0],
        vec![1.0],
        vec![0.0],
    ];

    let mut network = Network::new(vec![2, 3, 1], activation, activation_derivative, 0.5);
    network.train(inputs, targets, 10000);
    println!("0 XOR 0: {:?}", network.feedforward(Matrix::from(vec![0.0,0.0])));
    println!("0 XOR 1: {:?}", network.feedforward(Matrix::from(vec![0.0,1.0])));
    println!("1 XOR 0: {:?}", network.feedforward(Matrix::from(vec![1.0,0.0])));
    println!("1 XOR 1: {:?}", network.feedforward(Matrix::from(vec![1.0,1.0])));
}
