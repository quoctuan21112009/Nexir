fn compute(n: i64) -> i64 {
    let mut x: i64 = 1;
    for _ in 0..n {
        x = (x * 1_664_525 + 1_013_904_223) % 2_147_483_647;
    }
    x
}

fn main() {
    println!("{}", compute(20_000_000));
}
