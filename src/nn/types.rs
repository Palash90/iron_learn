use crate::numeric::FloatingPoint;

pub struct TrainingConfig<D> {
    pub epochs: usize,
    pub epoch_offset: usize,
    pub base_lr: D,
    pub lr_adjustment: bool,
}

pub struct TrainingHook<'a, F, S, D>
where
    F: FnMut(usize, D, D, &mut S),
    D: FloatingPoint,
{
    pub callback: F,
    pub interval: usize,
    _marker: std::marker::PhantomData<&'a S>,
    _type: std::marker::PhantomData<D>,
}

impl<'a, F, S, D> TrainingHook<'a, F, S, D>
where
    F: FnMut(usize, D, D, &mut S),
    D: FloatingPoint,
{
    pub fn new(interval: usize, callback: F) -> Self {
        Self {
            callback,
            interval,
            _marker: std::marker::PhantomData,
            _type: std::marker::PhantomData,
        }
    }
}
