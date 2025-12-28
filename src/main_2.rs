use rand::rngs::StdRng; // For Seedable range
use rand::Rng;
use rand::SeedableRng; // For Seedable range
use rand_distr::{Distribution, Normal, StandardNormal};
use rand_mt::Mt;
use std::f64::consts::PI;


fn main() {
    let mut rng = StdRng::seed_from_u64(1610612741);

    let mut mt_rng = Mt::new(1610612741);
    let normal = Normal::new(0.0, 1.0).unwrap();

    println!("Rust Generated sequence for f64");

    for _ in 0..5 {
        let sn: f64 = StandardNormal.sample(&mut rng);
        
        
        let normal = normal.sample(&mut rng);

        let rng_random = rng.random::<f64>();

        //let u1: f64 = mt_rng.gen();
        //let u2: f64 = mt_rng.gen();
        let mt = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();

        
        println!("sn: {sn},\tnormal: {normal},\trng.random: {rng_random},\tmt: {mt}");
    }
}
