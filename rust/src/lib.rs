
// Numerical simulation kernel; does not authenticate or authorize effects.

#[derive(Clone, Debug)]
pub struct State {
    pub load: Vec<f64>,
    pub capacity: Vec<f64>,
    pub edges: Vec<(usize, usize)>,
}

impl State {
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.load.is_empty() || self.load.len() != self.capacity.len() {
            return Err("shape");
        }
        for (&load, &capacity) in self.load.iter().zip(&self.capacity) {
            if !load.is_finite()
                || !capacity.is_finite()
                || capacity <= 0.0
                || load < 0.0
                || load > capacity
            {
                return Err("telemetry");
            }
        }
        let mut seen = std::collections::BTreeSet::new();
        for &(a, b) in &self.edges {
            if a >= b || b >= self.load.len() || !seen.insert((a, b)) {
                return Err("edge");
            }
        }
        Ok(())
    }

    pub fn curvature(&self) -> Result<Vec<f64>, &'static str> {
        self.validate()?;
        let mut result = vec![0.0; self.load.len()];
        for &(a, b) in &self.edges {
            let delta =
                self.load[b] / self.capacity[b]
                - self.load[a] / self.capacity[a];
            result[a] += delta;
            result[b] -= delta;
        }
        Ok(result)
    }

    pub fn energy(&self) -> Result<f64, &'static str> {
        self.validate()?;
        Ok(self.edges.iter().map(|&(a, b)| {
            let delta =
                self.load[a] / self.capacity[a]
                - self.load[b] / self.capacity[b];
            delta * delta / 2.0
        }).sum())
    }

    pub fn transfer(
        &self,
        source: usize,
        destination: usize,
        amount: f64,
        budget: f64,
    ) -> Result<Self, &'static str> {
        self.validate()?;
        if !budget.is_finite()
            || budget <= 0.0
            || !amount.is_finite()
            || amount <= 0.0
            || amount > budget
        {
            return Err("budget");
        }
        let edge = if source < destination {
            (source, destination)
        } else {
            (destination, source)
        };
        if !self.edges.contains(&edge) {
            return Err("edge permission");
        }
        let mut after = self.clone();
        after.load[source] -= amount;
        after.load[destination] += amount;
        after.validate()?;
        let before_mass: f64 = self.load.iter().sum();
        let after_mass: f64 = after.load.iter().sum();
        if (after_mass - before_mass).abs() > 1e-10 {
            return Err("conservation");
        }
        if after.energy()? > self.energy()? + 1e-12 {
            return Err("energy");
        }
        Ok(after)
    }
}

// Caller must supply valid, non-overlapping buffers of declared lengths.
#[no_mangle]
pub unsafe extern "C" fn gmc_curvature(
    n: usize,
    m: usize,
    load: *const f64,
    capacity: *const f64,
    edges: *const usize,
    output: *mut f64,
) -> i32 {
    if n == 0
        || n > 1_000_000
        || m > 4_000_000
        || load.is_null()
        || capacity.is_null()
        || output.is_null()
        || (m > 0 && edges.is_null())
    {
        return -1;
    }
    let edge_data = if m == 0 {
        &[][..]
    } else {
        std::slice::from_raw_parts(edges, 2 * m)
    };
    let state = State {
        load: std::slice::from_raw_parts(load, n).to_vec(),
        capacity: std::slice::from_raw_parts(capacity, n).to_vec(),
        edges: edge_data
            .chunks_exact(2)
            .map(|edge| (edge[0], edge[1]))
            .collect(),
    };
    match state.curvature() {
        Ok(values) => {
            std::ptr::copy_nonoverlapping(values.as_ptr(), output, n);
            0
        }
        Err(_) => -2,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state() -> State {
        State {
            load: vec![0.9, 0.1],
            capacity: vec![1.0, 1.0],
            edges: vec![(0, 1)],
        }
    }

    #[test]
    fn curvature_matches_reference() {
        assert_eq!(state().curvature().unwrap(), vec![-0.8, 0.8]);
    }

    #[test]
    fn transfer_reduces_energy() {
        let before = state();
        let after = before.transfer(0, 1, 0.05, 0.05).unwrap();
        assert!(after.energy().unwrap() < before.energy().unwrap());
    }

    #[test]
    fn nonfinite_transfer_is_refused() {
        assert!(state().transfer(0, 1, f64::NAN, 0.05).is_err());
    }

    #[test]
    fn contradictory_transfer_is_refused() {
        assert!(state().transfer(1, 0, 0.05, 0.05).is_err());
    }
}
