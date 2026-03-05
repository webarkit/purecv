use criterion::{black_box, criterion_group, criterion_main, Criterion};
use purecv::core::Matrix;
use purecv::core::arithm::add;

fn bench_add(c: &Criterion) {
    let size = 1024;
    let m1 = Matrix::<u8>::new(size, size, 3);
    let m2 = Matrix::<u8>::new(size, size, 3);

    c.bench_function("matrix_add_1024x1024x3", |b| {
        b.iter(|| add(black_box(&m1), black_box(&m2)).unwrap())
    });
}

criterion_group!(benches, bench_add);
criterion_main!(benches);
