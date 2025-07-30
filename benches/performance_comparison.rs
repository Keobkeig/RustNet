use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rust_neural_network::{Matrix, Network};

mod unoptimized {
    use rust_neural_network::Matrix;
    use std::ops::{Add, Mul, Sub};

    #[derive(Clone)]
    pub struct UnoptimizedMatrix {
        pub rows: usize,
        pub cols: usize,
        pub data: Vec<f64>,
    }

    impl From<Matrix> for UnoptimizedMatrix {
        fn from(m: Matrix) -> Self {
            UnoptimizedMatrix {
                rows: m.rows,
                cols: m.cols,
                data: m.data,
            }
        }
    }

    impl From<UnoptimizedMatrix> for Matrix {
        fn from(m: UnoptimizedMatrix) -> Self {
            Matrix {
                rows: m.rows,
                cols: m.cols,
                data: m.data,
            }
        }
    }

    impl Add for UnoptimizedMatrix {
        type Output = UnoptimizedMatrix;

        fn add(self, other: UnoptimizedMatrix) -> UnoptimizedMatrix {
            if self.rows != other.rows || self.cols != other.cols {
                panic!("Matrix dimensions must match for addition");
            }
            let data = self
                .data
                .iter()
                .zip(other.data.iter())
                .map(|(a, b)| a + b)
                .collect();
            UnoptimizedMatrix {
                rows: self.rows,
                cols: self.cols,
                data,
            }
        }
    }

    impl Sub for UnoptimizedMatrix {
        type Output = UnoptimizedMatrix;

        fn sub(self, other: UnoptimizedMatrix) -> UnoptimizedMatrix {
            if self.rows != other.rows || self.cols != other.cols {
                panic!("Matrix dimensions must match for subtraction");
            }
            let data = self
                .data
                .iter()
                .zip(other.data.iter())
                .map(|(a, b)| a - b)
                .collect();
            UnoptimizedMatrix {
                rows: self.rows,
                cols: self.cols,
                data,
            }
        }
    }

    impl Mul for UnoptimizedMatrix {
        type Output = UnoptimizedMatrix;

        fn mul(self, other: UnoptimizedMatrix) -> UnoptimizedMatrix {
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
            UnoptimizedMatrix {
                rows: self.rows,
                cols: other.cols,
                data: result,
            }
        }
    }

    pub struct UnoptimizedNetwork {
        layers: Vec<usize>,
        weights: Vec<UnoptimizedMatrix>,
        biases: Vec<UnoptimizedMatrix>,
        activations: Vec<UnoptimizedMatrix>,
        learning_rate: f64,
    }

    impl UnoptimizedNetwork {
        pub fn new(layers: Vec<usize>, learning_rate: f64) -> Self {
            let mut weights = Vec::new();
            let mut biases = Vec::new();

            for i in 0..layers.len() - 1 {
                let weight_matrix = Matrix::random(layers[i + 1], layers[i]);
                let bias_matrix = Matrix::random(layers[i + 1], 1);
                weights.push(UnoptimizedMatrix::from(weight_matrix));
                biases.push(UnoptimizedMatrix::from(bias_matrix));
            }

            UnoptimizedNetwork {
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

        pub fn feedforward_unoptimized(&mut self, input: UnoptimizedMatrix) -> UnoptimizedMatrix {
            self.activations.clear();
            self.activations.push(UnoptimizedMatrix {
                rows: input.rows,
                cols: input.cols,
                data: input.data.clone(),
            });

            let mut current = input;
            for i in 0..self.weights.len() {
                let weight_clone = UnoptimizedMatrix {
                    rows: self.weights[i].rows,
                    cols: self.weights[i].cols,
                    data: self.weights[i].data.clone(),
                };
                let current_clone = UnoptimizedMatrix {
                    rows: current.rows,
                    cols: current.cols,
                    data: current.data.clone(),
                };
                let bias_clone = UnoptimizedMatrix {
                    rows: self.biases[i].rows,
                    cols: self.biases[i].cols,
                    data: self.biases[i].data.clone(),
                };

                let z = weight_clone * current_clone + bias_clone;
                let activated_data = z.data.iter().map(|&x| Self::sigmoid(x)).collect();
                let activated = UnoptimizedMatrix {
                    rows: z.rows,
                    cols: z.cols,
                    data: activated_data,
                };

                self.activations.push(UnoptimizedMatrix {
                    rows: activated.rows,
                    cols: activated.cols,
                    data: activated.data.clone(), // Clone like the old version
                });
                current = activated;
            }
            current
        }
    }
}

fn bench_matrix_operations(c: &mut Criterion) {
    let size = 100;
    let a = Matrix::random(size, size);
    let b = Matrix::random(size, size);

    let a_unopt = unoptimized::UnoptimizedMatrix::from(a.clone());
    let b_unopt = unoptimized::UnoptimizedMatrix::from(b.clone());

    c.bench_function("optimized_matrix_multiply", |bench| {
        bench.iter(|| {
            let result = black_box(&a) * black_box(&b);
            black_box(result)
        })
    });

    c.bench_function("unoptimized_matrix_multiply", |bench| {
        bench.iter(|| {
            let a_clone = unoptimized::UnoptimizedMatrix::from(Matrix::from(a_unopt.clone()));
            let b_clone = unoptimized::UnoptimizedMatrix::from(Matrix::from(b_unopt.clone()));
            let result = black_box(a_clone) * black_box(b_clone);
            black_box(result)
        })
    });

    c.bench_function("optimized_matrix_add", |bench| {
        bench.iter(|| {
            let result = black_box(&a) + black_box(&b);
            black_box(result)
        })
    });

    c.bench_function("unoptimized_matrix_add", |bench| {
        bench.iter(|| {
            let a_clone = unoptimized::UnoptimizedMatrix::from(Matrix::from(a_unopt.clone()));
            let b_clone = unoptimized::UnoptimizedMatrix::from(Matrix::from(b_unopt.clone()));
            let result = black_box(a_clone) + black_box(b_clone);
            black_box(result)
        })
    });
}

fn bench_feedforward(c: &mut Criterion) {
    let mut optimized_network = Network::new(vec![10, 20, 10, 1], 0.1);
    let mut unoptimized_network = unoptimized::UnoptimizedNetwork::new(vec![10, 20, 10, 1], 0.1);

    let input = Matrix::random(10, 1);
    let input_unopt = unoptimized::UnoptimizedMatrix::from(input.clone());

    c.bench_function("optimized_feedforward", |bench| {
        bench.iter(|| {
            let result = optimized_network.feedforward(black_box(input.clone()));
            black_box(result)
        })
    });

    c.bench_function("unoptimized_feedforward", |bench| {
        bench.iter(|| {
            let input_copy =
                unoptimized::UnoptimizedMatrix::from(Matrix::from(input_unopt.clone()));
            let result = unoptimized_network.feedforward_unoptimized(black_box(input_copy));
            black_box(result)
        })
    });
}

fn bench_training_iteration(c: &mut Criterion) {
    let mut optimized_network = Network::new(vec![2, 4, 1], 0.5);
    let mut unoptimized_network = unoptimized::UnoptimizedNetwork::new(vec![2, 4, 1], 0.5);

    let inputs = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];
    let targets = vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]];

    c.bench_function("optimized_training_batch", |bench| {
        bench.iter(|| {
            for (inp, tar) in inputs.iter().zip(targets.iter()) {
                let input_matrix = Matrix::from_vec(inp.to_vec());
                let target_matrix = Matrix::from_vec(tar.to_vec());
                let _ = optimized_network.feedforward(black_box(input_matrix));
                optimized_network.backpropagate(black_box(target_matrix));
            }
        })
    });

    c.bench_function("unoptimized_training_batch", |bench| {
        bench.iter(|| {
            for (inp, _tar) in inputs.iter().zip(targets.iter()) {
                let input_matrix =
                    unoptimized::UnoptimizedMatrix::from(Matrix::from_vec(inp.clone()));
                let _ = unoptimized_network.feedforward_unoptimized(black_box(input_matrix));
                // Note: We're only benchmarking feedforward for the unoptimized version
            }
        })
    });
}

criterion_group!(
    benches,
    bench_matrix_operations,
    bench_feedforward,
    bench_training_iteration
);
criterion_main!(benches);
