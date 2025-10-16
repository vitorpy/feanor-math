use std::alloc::{Allocator, Global};
use std::cell::{RefCell, RefMut};
use std::fmt::Debug;

use atomicbox::AtomicOptionBox;
use thread_local::ThreadLocal;

use crate::iters::multiset_combinations;
use crate::primitive_int::StaticRing;
use crate::ring::*;
use crate::rings::multivariate::*;
use crate::integer::*;
use crate::seq::{VectorFn, VectorView};
use crate::homomorphism::*;
use crate::ordered::OrderedRingStore;

type Exponent = u16;
type OrderIdx = u64;

///
/// Error type for monomial degree overflow.
///
/// This error occurs when a monomial operation (e.g., multiplication, lcm) would produce
/// a degree exceeding the ring's `max_supported_deg` configuration parameter.
///
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DegreeOverflowError {
    pub max_supported_deg: Exponent,
    pub attempted_deg: Exponent,
    pub operation: &'static str,
}

impl std::fmt::Display for DegreeOverflowError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Degree overflow in {}: ring supports max degree {}, but operation produced degree {}",
            self.operation, self.max_supported_deg, self.attempted_deg
        )
    }
}

impl std::error::Error for DegreeOverflowError {}

///
/// Monomial indexing mode for polynomial rings.
///
/// Controls how monomials are stored and accessed internally:
/// - `Auto`: Automatically choose Dense for low degrees, Sparse for high degrees
/// - `DenseOnly`: Always use dense (indexed) mode; panic if degree tables overflow
/// - `SparseOnly`: Always use sparse (explicit exponent) mode
///
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IndexMode {
    /// Automatically choose between Dense and Sparse based on safe_degree_threshold
    Auto,
    /// Always use dense indexing; panic on overflow
    DenseOnly,
    /// Always use sparse (exponent vector) indexing
    SparseOnly,
}

impl Default for IndexMode {
    fn default() -> Self {
        IndexMode::Auto
    }
}

///
/// Degree configuration for polynomial rings.
///
/// Specifies the maximum supported degree and the degree up to which precomputed
/// tables (binomial coefficients, multiplication tables) should be built.
///
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DegreeCfg {
    /// Maximum degree that monomials can reach; operations exceeding this will error
    pub max_supported_deg: u16,
    /// Maximum degree for which to precompute lookup tables; higher degrees use on-demand computation
    pub max_precompute_deg: Option<u16>,
}

impl DegreeCfg {
    /// Create a new degree configuration
    pub fn new(max_supported_deg: u16) -> Self {
        DegreeCfg {
            max_supported_deg,
            max_precompute_deg: None, // Auto-determine based on variable count
        }
    }

    /// Set the maximum degree for precomputed tables
    pub fn with_precompute(mut self, max_precompute_deg: u16) -> Self {
        self.max_precompute_deg = Some(max_precompute_deg);
        self
    }
}

///
/// Error type for monomial indexing failures.
///
/// This error occurs when dense indexing is explicitly required (IndexMode::DenseOnly)
/// but the degree would exceed the safe threshold.
///
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MonomialIndexError {
    pub degree: u16,
    pub safe_threshold: u16,
    pub indexing_mode: &'static str,
}

impl std::fmt::Display for MonomialIndexError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Monomial indexing error: degree {} exceeds safe threshold {} for {} mode",
            self.degree, self.safe_threshold, self.indexing_mode
        )
    }
}

impl std::error::Error for MonomialIndexError {}

///
/// Computes the "cumulative binomial function" `sum_(0 <= l <= k) binomial(n + l, n)`
/// 
fn compute_cum_binomial(n: usize, k: usize) -> u64 {
    StaticRing::<i64>::RING.sum((0..(k + 1)).map(|l| binomial((n + l) as i128, &(n as i128), StaticRing::<i128>::RING).try_into().unwrap())) as u64
}

///
/// Returns the index of the given monomial within the list of all degree-d monomials, ordered by DegRevLex
/// 
fn enumeration_index_degrevlex<V>(d: Exponent, mon: V, cum_binomial_lookup_table: &[Vec<u64>]) -> u64
    where V: VectorFn<Exponent>
{
    debug_assert!(d == mon.iter().sum::<Exponent>());
    let n = mon.len();
    let mut remaining_degree_minus_one: i64 = TryInto::<i64>::try_into(d).unwrap() - 1;
    let mut result = 0;
    for i in 0..(n - 1) {
        remaining_degree_minus_one -= TryInto::<i64>::try_into(mon.at(n - 1 - i)).unwrap();
        if remaining_degree_minus_one < 0 {
            return result;
        }
        result += cum_binomial_lookup_table[n - i - 2][remaining_degree_minus_one as usize];
    }
    return result;
}

///
/// Inverse to [`enumeration_index_degrevlex()`].
/// 
fn nth_monomial_degrevlex<F>(n: usize, d: Exponent, mut index: u64, cum_binomial_lookup_table: &[Vec<u64>], mut out: F)
    where F: FnMut(usize, Exponent)
{
    for i in 0..n {
        out(i, 0);
    }
    let mut check_degree = 0;
    let mut remaining_degree = d as usize;
    for i in 0..(n - 1) {
        if index == 0 {
            out(n - 1 - i, remaining_degree as Exponent);
            check_degree += remaining_degree as Exponent;
            debug_assert!(d == check_degree);
            return;
        }
        let remaining_degree_minus_one = match cum_binomial_lookup_table[n - i - 2].binary_search(&index) {
            Ok(idx) => idx,
            Err(idx) => idx - 1
        };
        index -= cum_binomial_lookup_table[n - i - 2][remaining_degree_minus_one];
        let new_remaining_degree = remaining_degree_minus_one + 1;
        out(n - 1 - i, (remaining_degree - new_remaining_degree) as Exponent);
        check_degree += (remaining_degree - new_remaining_degree) as Exponent;
        remaining_degree = new_remaining_degree;
    }
    out(0, remaining_degree as Exponent);
    check_degree += remaining_degree as Exponent;
    debug_assert!(d == check_degree);
}

///
/// Stores a reference to a monomial w.r.t. a given [`MultivariatePolyRingImplBase`].
///
#[repr(transparent)]
pub struct MonomialIdentifier {
    data: InternalMonomialIdentifier
}

impl Debug for MonomialIdentifier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.data.data {
            InternalMonomialData::Dense { deg, order } => {
                f.debug_struct("MonomialIdentifier")
                    .field("deg", deg)
                    .field("idx", order)
                    .finish()
            }
            InternalMonomialData::Sparse { deg, .. } => {
                f.debug_struct("MonomialIdentifier")
                    .field("deg", deg)
                    .field("mode", &"sparse")
                    .finish()
            }
        }
    }
}

///
/// Internal representation supporting both dense (indexed) and sparse (explicit exponents) modes.
/// Dense mode uses lookup tables for fast operations when per-degree monomial counts fit in u64.
/// Sparse mode stores exponents directly for high degrees where enumeration would overflow.
///
#[derive(Clone, Debug)]
enum InternalMonomialData {
    Dense {
        deg: Exponent,
        order: OrderIdx,
    },
    Sparse {
        deg: Exponent,
        exponents: Box<[Exponent]>,
    },
}

#[derive(Clone, Debug)]
struct InternalMonomialIdentifier {
    data: InternalMonomialData,
}

impl InternalMonomialIdentifier {

    fn wrap(self) -> MonomialIdentifier {
        MonomialIdentifier { data: self }
    }

    fn deg(&self) -> Exponent {
        match &self.data {
            InternalMonomialData::Dense { deg, .. } => *deg,
            InternalMonomialData::Sparse { deg, .. } => *deg,
        }
    }
}

impl PartialEq for InternalMonomialIdentifier {
    fn eq(&self, other: &Self) -> bool {
        match (&self.data, &other.data) {
            (InternalMonomialData::Dense { deg: d1, order: o1 }, InternalMonomialData::Dense { deg: d2, order: o2 }) => {
                d1 == d2 && o1 == o2
            }
            (InternalMonomialData::Sparse { deg: d1, exponents: e1 }, InternalMonomialData::Sparse { deg: d2, exponents: e2 }) => {
                d1 == d2 && e1 == e2
            }
            _ => false, // Dense != Sparse
        }
    }
}

impl Eq for InternalMonomialIdentifier {}

impl Ord for InternalMonomialIdentifier {
    fn cmp(&self, other: &Self) -> Ordering {
        // DegRevLex order: first by degree, then by lexicographic comparison (reversed)
        // For now, delegate to detailed comparison; this will be refined below
        self.deg().cmp(&other.deg()).then_with(|| {
            // If both are dense, use order index
            match (&self.data, &other.data) {
                (InternalMonomialData::Dense { order: o1, .. }, InternalMonomialData::Dense { order: o2, .. }) => {
                    o1.cmp(o2)
                }
                // For sparse or mixed, need exponent-wise comparison (implemented per-ring below)
                _ => Ordering::Equal, // Placeholder; actual comparison done in ring methods
            }
        })
    }
}

impl PartialOrd for InternalMonomialIdentifier {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

///
/// An element of [`MultivariatePolyRingImpl`].
/// 
pub struct MultivariatePolyRingEl<R, A = Global>
    where R: RingStore,
        A: Allocator + Clone
{
    data: Vec<(El<R>, MonomialIdentifier), A>
}

impl<R, A> Debug for MultivariatePolyRingEl<R, A>
    where R: RingStore,
        El<R>: Debug,
        A: Allocator + Clone
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.data)
    }
}

///
/// Implementation of multivariate polynomial rings.
///
pub struct MultivariatePolyRingImplBase<R, A = Global>
    where R: RingStore,
        A: Clone + Allocator + Send
{
    base_ring: R,
    variable_count: usize,
    /// why do I use a RefCell here? Note that in `create_monomial`, we are putting values into `tmp_monomial`
    /// from an iterator, which means that theoretically, the `next()` of the iterator might call `create_monomial()`
    /// again. Using a `RefCell` at least leads to a panic in this crazy unlikely scenario, while using `Cell` would
    /// silently give wrong results.
    tmp_monomials: ThreadLocal<(RefCell<Box<[Exponent]>>, RefCell<Box<[Exponent]>>)>,
    /// indices are [lhs_deg][rhs_deg][lhs_index][rhs_index]
    monomial_multiplication_table: Vec<Vec<Vec<Vec<u64>>>>,
    zero: El<R>,
    cum_binomial_lookup_table: Vec<Vec<u64>>,
    max_supported_deg: Exponent,
    /// Maximum degree for which dense (indexed) monomials can be used.
    /// For degrees > safe_degree_threshold, sparse mode is automatically used.
    safe_degree_threshold: Exponent,
    allocator: A,
    tmp_poly: AtomicOptionBox<Vec<(El<R>, MonomialIdentifier)>>
}

///
/// [`RingStore`] corresponding to [`MultivariatePolyRingImplBase`]
/// 
pub type MultivariatePolyRingImpl<R, A = Global> = RingValue<MultivariatePolyRingImplBase<R, A>>;

impl<R> MultivariatePolyRingImpl<R>
    where R: RingStore
{
    ///
    /// Creates a new instance of the ring `base_ring[X0, ..., Xn]` where `n = variable_count - 1`.
    /// 
    pub fn new(base_ring: R, variable_count: usize) -> Self {
        Self::new_with_mult_table(base_ring, variable_count, 64, (6, 8), Global)
    }
}

impl<R, A> MultivariatePolyRingImpl<R, A>
    where R: RingStore,
        A: Clone + Allocator + Send
{
    ///
    /// Creates a new instance of the ring `base_ring[X0, ..., Xn]` where `n = variable_count - 1`.
    /// 
    /// The can represent all monomials up to the given degree, and will panic should an operation
    /// produce a monomial that exceeds this degree. 
    /// 
    /// Furthermore, `max_multiplication_table = (d1, d2)` configures for which monomials a multiplication
    /// table is precomputed. In particular, a multiplication table is precomputed for all products where
    /// one summand has degree `<= d2` and the other summand has degree `<= d1`. Note that `d1 <= d2` is
    /// required.
    ///
    pub fn new_with_mult_table(base_ring: R, variable_count: usize, max_supported_deg: Exponent, max_multiplication_table: (Exponent, Exponent), allocator: A) -> Self {
        assert!(variable_count >= 1);
        assert!(max_multiplication_table.0 <= max_multiplication_table.1);
        assert!(max_multiplication_table.0 + max_multiplication_table.1 <= max_supported_deg);

        // RELAXED CONSTRAINT: Check per-degree monomial counts instead of cumulative.
        //
        // Context: This change was made to support the zyga ZKP subsystem slicer use case,
        // which needs 64 variables with max degree 2. The original check was overly conservative,
        // requiring that the *cumulative* number of monomials across all degrees fit in a u64.
        //
        // The actual requirement is that for each degree d, the OrderIdx (u64) can represent
        // the index of any monomial of degree d within the set of all degree-d monomials.
        // The number of degree-d monomials in n variables is binomial(n + d - 1, d).
        //
        // For 64 variables, degree 2: binomial(65, 2) = 2,080 << u64::MAX.
        // Total monomials (deg 0-2): 1 + 64 + 2,080 = 2,145 (easily fits).
        //
        // Safety invariants that must hold:
        // - OrderIdx (u64) can represent any monomial index at each individual degree
        // - The cum_binomial_lookup_table values (used for monomial enumeration) fit in u64
        // - Monomial multiplication table construction doesn't overflow
        //
        // We now check that each individual degree's monomial count fits in u64, rather than
        // checking an overly conservative cumulative bound.

        // Compute safe_degree_threshold: the highest degree for which per-degree monomial
        // count fits in u64. For degrees above this, we'll use sparse mode.
        let mut safe_degree_threshold = max_supported_deg;
        for d in 0..=max_supported_deg {
            let monomial_count = binomial(
                int_cast((variable_count + d as usize - 1) as i128, BigIntRing::RING, StaticRing::<i128>::RING),
                &int_cast(d as i128, BigIntRing::RING, StaticRing::<i128>::RING),
                BigIntRing::RING
            );
            if !BigIntRing::RING.is_lt(&monomial_count, &BigIntRing::RING.power_of_two(u64::BITS as usize)) {
                // This degree exceeds u64, so safe threshold is d-1
                safe_degree_threshold = d.saturating_sub(1);
                break;
            }
        }

        // Build lookup tables only up to safe_degree_threshold
        // This ensures cum_binomial values won't overflow
        let table_max_deg = safe_degree_threshold;

        // Verify that cum_binomial values for the lookup table will fit in u64
        if variable_count > 2 && table_max_deg > 0 {
            let max_cum_binomial = BigIntRing::RING.sum(
                (0..=table_max_deg).map(|l| {
                    binomial(
                        int_cast((variable_count - 2 + l as usize) as i128, BigIntRing::RING, StaticRing::<i128>::RING),
                        &int_cast((variable_count - 2) as i128, BigIntRing::RING, StaticRing::<i128>::RING),
                        BigIntRing::RING
                    )
                })
            );
            assert!(
                BigIntRing::RING.is_lt(&max_cum_binomial, &BigIntRing::RING.power_of_two(u64::BITS as usize)),
                "Cumulative binomial lookup table entry would exceed u64::MAX at safe_degree_threshold {}. This is a bug.", safe_degree_threshold
            );
        }

        let cum_binomial_lookup_table = (0..(variable_count - 1)).map(|n| (0..=table_max_deg).map(|k| compute_cum_binomial(n, k as usize)).collect::<Vec<_>>()).collect::<Vec<_>>();

        // Multiplication tables: only build for degrees within safe threshold
        let mult_table_max = (max_multiplication_table.0.min(safe_degree_threshold), max_multiplication_table.1.min(safe_degree_threshold));
        let monomial_multipliation_table = (0..=mult_table_max.0).map(|lhs_deg| (lhs_deg..=mult_table_max.1).map(|rhs_deg| MultivariatePolyRingImplBase::<R, A>::create_multiplication_table(variable_count, lhs_deg, rhs_deg, &cum_binomial_lookup_table)).collect::<Vec<_>>()).collect::<Vec<_>>();

        RingValue::from(MultivariatePolyRingImplBase {
            zero: base_ring.zero(),
            base_ring: base_ring,
            variable_count: variable_count,
            max_supported_deg: max_supported_deg,
            safe_degree_threshold: safe_degree_threshold,
            monomial_multiplication_table: monomial_multipliation_table,
            tmp_monomials: ThreadLocal::new(),
            cum_binomial_lookup_table: cum_binomial_lookup_table,
            tmp_poly: AtomicOptionBox::none(),
            allocator: allocator,
        })
    }

    ///
    /// Extended constructor for large-variable systems with explicit degree configuration.
    ///
    /// This constructor provides more control over degree handling and table precomputation,
    /// which is essential for systems with many variables (e.g., 64+ variables) where
    /// cumulative degree tables would overflow.
    ///
    /// # Parameters
    /// - `base_ring`: The coefficient ring
    /// - `variable_count`: Number of indeterminates
    /// - `degree_cfg`: Degree limits and precomputation configuration
    /// - `max_multiplication_table`: Multiplication table precomputation limits (d1, d2)
    /// - `allocator`: Memory allocator
    ///
    /// # Example
    /// ```ignore
    /// use feanor_math::rings::multivariate::multivariate_impl::{MultivariatePolyRingImpl, DegreeCfg};
    /// use feanor_math::primitive_int::StaticRing;
    /// use std::alloc::Global;
    ///
    /// // 64 variables, max degree 30, precompute tables only up to degree 10
    /// let degree_cfg = DegreeCfg::new(30).with_precompute(10);
    /// let ring = MultivariatePolyRingImpl::new_with_mult_table_ex(
    ///     StaticRing::<i64>::RING,
    ///     64,
    ///     degree_cfg,
    ///     (2, 2),
    ///     Global
    /// );
    /// ```
    pub fn new_with_mult_table_ex(
        base_ring: R,
        variable_count: usize,
        degree_cfg: DegreeCfg,
        max_multiplication_table: (Exponent, Exponent),
        allocator: A
    ) -> Self {
        assert!(variable_count >= 1);
        assert!(max_multiplication_table.0 <= max_multiplication_table.1);
        assert!(max_multiplication_table.0 + max_multiplication_table.1 <= degree_cfg.max_supported_deg);

        let max_supported_deg = degree_cfg.max_supported_deg;

        // Compute safe_degree_threshold: highest degree where per-degree monomial count fits in u64
        let mut safe_degree_threshold = max_supported_deg;
        for d in 0..=max_supported_deg {
            let monomial_count = binomial(
                int_cast((variable_count + d as usize - 1) as i128, BigIntRing::RING, StaticRing::<i128>::RING),
                &int_cast(d as i128, BigIntRing::RING, StaticRing::<i128>::RING),
                BigIntRing::RING
            );
            if !BigIntRing::RING.is_lt(&monomial_count, &BigIntRing::RING.power_of_two(u64::BITS as usize)) {
                safe_degree_threshold = d.saturating_sub(1);
                break;
            }
        }

        // Determine table_max_deg: limit table building based on max_precompute_deg if specified
        let table_max_deg = if let Some(max_precompute) = degree_cfg.max_precompute_deg {
            max_precompute.min(safe_degree_threshold)
        } else {
            safe_degree_threshold
        };

        // Verify cum_binomial values won't overflow
        if variable_count > 2 && table_max_deg > 0 {
            let max_cum_binomial = BigIntRing::RING.sum(
                (0..=table_max_deg).map(|l| {
                    binomial(
                        int_cast((variable_count - 2 + l as usize) as i128, BigIntRing::RING, StaticRing::<i128>::RING),
                        &int_cast((variable_count - 2) as i128, BigIntRing::RING, StaticRing::<i128>::RING),
                        BigIntRing::RING
                    )
                })
            );
            assert!(
                BigIntRing::RING.is_lt(&max_cum_binomial, &BigIntRing::RING.power_of_two(u64::BITS as usize)),
                "Cumulative binomial lookup table entry would exceed u64::MAX at table_max_deg {}. This is a bug.", table_max_deg
            );
        }

        let cum_binomial_lookup_table = (0..(variable_count - 1))
            .map(|n| (0..=table_max_deg)
                .map(|k| compute_cum_binomial(n, k as usize))
                .collect::<Vec<_>>())
            .collect::<Vec<_>>();

        // Build multiplication tables only up to the smaller of table_max_deg and requested limits
        let mult_table_max = (
            max_multiplication_table.0.min(table_max_deg),
            max_multiplication_table.1.min(table_max_deg)
        );

        let monomial_multipliation_table = (0..=mult_table_max.0)
            .map(|lhs_deg| (lhs_deg..=mult_table_max.1)
                .map(|rhs_deg| MultivariatePolyRingImplBase::<R, A>::create_multiplication_table(
                    variable_count, lhs_deg, rhs_deg, &cum_binomial_lookup_table
                ))
                .collect::<Vec<_>>())
            .collect::<Vec<_>>();

        RingValue::from(MultivariatePolyRingImplBase {
            zero: base_ring.zero(),
            base_ring: base_ring,
            variable_count: variable_count,
            max_supported_deg: max_supported_deg,
            safe_degree_threshold: safe_degree_threshold,
            monomial_multiplication_table: monomial_multipliation_table,
            tmp_monomials: ThreadLocal::new(),
            cum_binomial_lookup_table: cum_binomial_lookup_table,
            tmp_poly: AtomicOptionBox::none(),
            allocator: allocator,
        })
    }
}

impl<R, A> MultivariatePolyRingImplBase<R, A>
    where R: RingStore,
        A: Clone + Allocator + Send
{
    pub fn allocator(&self) -> &A {
        &self.allocator
    }

    #[inline(always)]
    fn tmp_monomials(&self) -> (RefMut<'_, [u16]>, RefMut<'_, [u16]>) {
        (self.tmp_monomial1(), self.tmp_monomial2())
    }

    ///
    /// We have to temporary monomials that are used as follows:
    ///  - `tmp_monomial2` for `create_monomial()`
    ///  - both for bivariate operations in monomials (e.g. `monomial_mul()`)
    /// 
    /// We use `#[inline(always)]` in the hope that the compiler can optimize out runtime 
    /// checks of `RefCell`.
    /// 
    #[inline(always)]
    fn tmp_monomial1(&self) -> RefMut<'_, [u16]> {
        let (mon, _) = self.tmp_monomials.get_or(|| (RefCell::new((0..self.variable_count).map(|_| 0).collect::<Vec<_>>().into_boxed_slice()), RefCell::new((0..self.variable_count).map(|_| 0).collect::<Vec<_>>().into_boxed_slice())));
        RefMut::map(mon.borrow_mut(), |mon| &mut**mon)
    }

    ///
    /// See [`MultivariatePolyRingImplBase::tmp_monomial1()`]
    /// 
    #[inline(always)]
    fn tmp_monomial2(&self) -> RefMut<'_, [u16]> {
        let (_, mon) = self.tmp_monomials.get_or(|| (RefCell::new((0..self.variable_count).map(|_| 0).collect::<Vec<_>>().into_boxed_slice()), RefCell::new((0..self.variable_count).map(|_| 0).collect::<Vec<_>>().into_boxed_slice())));
        RefMut::map(mon.borrow_mut(), |mon| &mut**mon)
    }

    #[inline(always)]
    fn exponent_wise_bivariate_monomial_operation<F>(&self, lhs: InternalMonomialIdentifier, rhs: InternalMonomialIdentifier, mut f: F) -> MonomialIdentifier
        where F: FnMut(Exponent, Exponent) -> Exponent
    {
        let (mut lhs_mon, mut rhs_mon) = self.tmp_monomials();

        // Extract exponents from lhs
        match &lhs.data {
            InternalMonomialData::Dense { deg, order } => {
                nth_monomial_degrevlex(self.variable_count, *deg, *order, &self.cum_binomial_lookup_table, |i, x| lhs_mon[i] = x);
            }
            InternalMonomialData::Sparse { exponents, .. } => {
                for (i, &e) in exponents.iter().enumerate() {
                    lhs_mon[i] = e;
                }
            }
        }

        // Extract exponents from rhs
        match &rhs.data {
            InternalMonomialData::Dense { deg, order } => {
                nth_monomial_degrevlex(self.variable_count, *deg, *order, &self.cum_binomial_lookup_table, |i, x| rhs_mon[i] = x);
            }
            InternalMonomialData::Sparse { exponents, .. } => {
                for (i, &e) in exponents.iter().enumerate() {
                    rhs_mon[i] = e;
                }
            }
        }

        // Apply operation and compute result
        let mut res_deg = 0;
        for i in 0..self.variable_count {
            lhs_mon[i] = f(lhs_mon[i], rhs_mon[i]);
            res_deg += lhs_mon[i];
        }
        assert!(res_deg <= self.max_supported_deg, "Polynomial ring was configured to support monomials up to degree {}, but operation resulted in degree {}", self.max_supported_deg, res_deg);

        // Create result monomial (dense or sparse based on degree)
        if res_deg <= self.safe_degree_threshold {
            return MonomialIdentifier {
                data: InternalMonomialIdentifier {
                    data: InternalMonomialData::Dense {
                        deg: res_deg,
                        order: enumeration_index_degrevlex(res_deg, (&*lhs_mon).clone_els_by(|x| *x), &self.cum_binomial_lookup_table)
                    }
                }
            };
        } else {
            let exponents_vec: Vec<Exponent> = lhs_mon.iter().cloned().collect();
            return MonomialIdentifier {
                data: InternalMonomialIdentifier {
                    data: InternalMonomialData::Sparse {
                        deg: res_deg,
                        exponents: exponents_vec.into_boxed_slice()
                    }
                }
            };
        }
    }

    fn create_multiplication_table(variable_count: usize, lhs_deg: Exponent, rhs_deg: Exponent, cum_binomial_lookup_table: &[Vec<u64>]) -> Vec<Vec<u64>> {
        debug_assert!(lhs_deg <= rhs_deg);
        let lhs_max_deg = (0..variable_count).map(|_| lhs_deg as usize).collect::<Vec<_>>();
        let rhs_max_deg = (0..variable_count).map(|_| rhs_deg as usize).collect::<Vec<_>>();
        let mut lhs_i = 0;
        let mut rhs_i = 0;
        // `multiset_combinations()` iterates through the values in descending lex order, so by reversing the variable order, we get degrevlex
        let rev = |i| variable_count - 1 - i;
        multiset_combinations(&lhs_max_deg, lhs_deg as usize, |lhs| {
            let result = multiset_combinations(&rhs_max_deg, rhs_deg as usize, |rhs| {
                let result_index = enumeration_index_degrevlex(lhs_deg + rhs_deg, (0..variable_count).map_fn(|i| (lhs[rev(i)] + rhs[rev(i)]) as u16), cum_binomial_lookup_table);
                rhs_i += 1;
                return result_index;
            }).collect::<Vec<_>>();
            lhs_i += 1;
            return result;
        }).collect::<Vec<_>>()
    }

    fn try_get_multiplication_table<'a>(&'a self, lhs_deg: Exponent, rhs_deg: Exponent) -> Option<&'a Vec<Vec<u64>>> {
        debug_assert!(lhs_deg <= rhs_deg);
        if lhs_deg as usize >= self.monomial_multiplication_table.len() || rhs_deg as usize >= self.monomial_multiplication_table[lhs_deg as usize].len() {
            return None;
        }
        Some(&self.monomial_multiplication_table[lhs_deg as usize][rhs_deg as usize])
    }

    fn compare_degrevlex(&self, lhs: &InternalMonomialIdentifier, rhs: &InternalMonomialIdentifier) -> Ordering {
        // First compare by degree
        let res = lhs.deg().cmp(&rhs.deg()).then_with(|| {
            // If degrees are equal, compare based on mode
            match (&lhs.data, &rhs.data) {
                // Both dense: use order index
                (InternalMonomialData::Dense { order: o1, .. }, InternalMonomialData::Dense { order: o2, .. }) => {
                    o1.cmp(o2)
                }
                // Both sparse or mixed: need exponent-wise degrevlex comparison
                _ => {
                    // DegRevLex: reverse lexicographic order on exponents
                    // Compare from last variable to first (reversed)
                    let mut lhs_mon = self.tmp_monomial1();
                    match &lhs.data {
                        InternalMonomialData::Dense { deg, order } => {
                            nth_monomial_degrevlex(self.variable_count, *deg, *order, &self.cum_binomial_lookup_table, |i, x| lhs_mon[i] = x);
                        }
                        InternalMonomialData::Sparse { exponents, .. } => {
                            for (i, &e) in exponents.iter().enumerate() {
                                lhs_mon[i] = e;
                            }
                        }
                    }

                    let mut rhs_mon = self.tmp_monomial2();
                    match &rhs.data {
                        InternalMonomialData::Dense { deg, order } => {
                            nth_monomial_degrevlex(self.variable_count, *deg, *order, &self.cum_binomial_lookup_table, |i, x| rhs_mon[i] = x);
                        }
                        InternalMonomialData::Sparse { exponents, .. } => {
                            for (i, &e) in exponents.iter().enumerate() {
                                rhs_mon[i] = e;
                            }
                        }
                    }

                    // Reverse lexicographic comparison
                    for i in (0..self.variable_count).rev() {
                        match lhs_mon[i].cmp(&rhs_mon[i]) {
                            Ordering::Less => return Ordering::Greater,
                            Ordering::Greater => return Ordering::Less,
                            Ordering::Equal => continue,
                        }
                    }
                    Ordering::Equal
                }
            }
        });
        debug_assert!(res == DegRevLex.compare(RingRef::new(self), &lhs.clone().wrap(), &rhs.clone().wrap()));
        return res;
    }

    ///
    /// Compare two monomials under Lex (lexicographic) ordering.
    ///
    /// Lex ordering compares exponents variable by variable from first to last:
    /// X0 > X1 > X2 > ... This method is optimized for sparse monomials, which
    /// is crucial for 64-variable elimination systems.
    ///
    fn compare_lex(&self, lhs: &InternalMonomialIdentifier, rhs: &InternalMonomialIdentifier) -> Ordering {
        // Fast path: both sparse - directly compare exponent arrays
        if let (InternalMonomialData::Sparse { exponents: lhs_exp, .. },
                 InternalMonomialData::Sparse { exponents: rhs_exp, .. }) = (&lhs.data, &rhs.data) {
            // Lexicographic comparison: compare from first to last variable
            for i in 0..self.variable_count {
                match lhs_exp[i].cmp(&rhs_exp[i]) {
                    Ordering::Greater => return Ordering::Greater,
                    Ordering::Less => return Ordering::Less,
                    Ordering::Equal => continue,
                }
            }
            return Ordering::Equal;
        }

        // Slower path: need to extract exponents for at least one monomial
        let mut lhs_mon = self.tmp_monomial1();
        match &lhs.data {
            InternalMonomialData::Dense { deg, order } => {
                nth_monomial_degrevlex(self.variable_count, *deg, *order, &self.cum_binomial_lookup_table, |i, x| lhs_mon[i] = x);
            }
            InternalMonomialData::Sparse { exponents, .. } => {
                for (i, &e) in exponents.iter().enumerate() {
                    lhs_mon[i] = e;
                }
            }
        }

        let mut rhs_mon = self.tmp_monomial2();
        match &rhs.data {
            InternalMonomialData::Dense { deg, order } => {
                nth_monomial_degrevlex(self.variable_count, *deg, *order, &self.cum_binomial_lookup_table, |i, x| rhs_mon[i] = x);
            }
            InternalMonomialData::Sparse { exponents, .. } => {
                for (i, &e) in exponents.iter().enumerate() {
                    rhs_mon[i] = e;
                }
            }
        }

        // Lexicographic comparison: compare from first to last variable
        for i in 0..self.variable_count {
            match lhs_mon[i].cmp(&rhs_mon[i]) {
                Ordering::Greater => return Ordering::Greater,
                Ordering::Less => return Ordering::Less,
                Ordering::Equal => continue,
            }
        }
        Ordering::Equal
    }

    fn is_valid(&self, el: &[(El<R>, MonomialIdentifier)]) -> bool {
        for i in 1..el.len() {
            if self.compare_degrevlex(&el[i - 1].1.data, &el[i].1.data) != Ordering::Less {
                return false;
            }
        }
        if !self.base_ring().get_ring().is_approximate() {
            for i in 0..el.len() {
                if self.base_ring().is_zero(&el[i].0) {
                    return false;
                }
            }
        }
        return true;
    }

    fn remove_zeros(&self, el: &mut Vec<(El<R>, MonomialIdentifier), A>) {
        if self.base_ring().get_ring().is_approximate() {
            return;
        }
        el.retain(|(c, _)| !self.base_ring().is_zero(c));
        debug_assert!(self.is_valid(&el));
    }

    ///
    /// Computes the sum of two elements; rhs may contain zero elements, but must be sorted and not contain equal monomials
    ///
    fn add_terms<I>(&self, lhs: &<Self as RingBase>::Element, rhs_sorted: I, out: Vec<(El<R>, MonomialIdentifier), A>) -> <Self as RingBase>::Element
        where I: Iterator<Item = (El<R>, MonomialIdentifier)>
    {
        debug_assert!(self.is_valid(&lhs.data));
        
        let mut result = out;
        result.clear();
        result.reserve(lhs.data.len() + rhs_sorted.size_hint().0);
        
        let mut lhs_it = lhs.data.iter().peekable();
        let mut rhs_it = rhs_sorted.peekable();

        while let (Some((_, l_m)), Some((_, r_m))) = (lhs_it.peek(), rhs_it.peek()) {
            let next_element = match self.compare_degrevlex(&l_m.data, &r_m.data) {
                Ordering::Equal => {
                    let (l_c, _l_m) = lhs_it.next().unwrap();
                    let (r_c, r_m) = rhs_it.next().unwrap();
                    (self.base_ring().add_ref_fst(l_c, r_c), r_m)
                },
                Ordering::Less => {
                    let (l_c, l_m) = lhs_it.next().unwrap();
                    (self.base_ring().clone_el(l_c), l_m.data.clone().wrap())
                },
                Ordering::Greater => {
                    let (r_c, r_m) = rhs_it.next().unwrap();
                    (r_c, r_m)
                }
            };
            result.push(next_element);
        }
        result.extend(lhs_it.map(|(c, m)| (self.base_ring().clone_el(c), m.data.clone().wrap())));
        result.extend(rhs_it);
        self.remove_zeros(&mut result);
        debug_assert!(self.is_valid(&result));
        return MultivariatePolyRingEl {
            data: result
        };
    }
}

impl<R, A> Debug for MultivariatePolyRingImplBase<R, A>
    where R: RingStore,
        R::Type: Debug,
        A: Clone + Allocator + Send
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MultivariatePolyRingImplBase")
            .field("base_ring", &self.base_ring.get_ring())
            .field("variable_count", &self.variable_count)
            .field("max_supported_deg", &self.max_supported_deg)
            .finish()
    }
}

impl<R, A> PartialEq for MultivariatePolyRingImplBase<R, A>
    where R: RingStore,
        A: Clone + Allocator + Send
{
    fn eq(&self, other: &Self) -> bool {
        self.base_ring.get_ring() == other.base_ring.get_ring() && self.variable_count == other.variable_count && self.max_supported_deg == other.max_supported_deg
    }
}

impl<R, A> RingBase for MultivariatePolyRingImplBase<R, A>
    where R: RingStore,
        A: Clone + Allocator + Send
{
    type Element = MultivariatePolyRingEl<R, A>;

    fn clone_el(&self, val: &Self::Element) -> Self::Element {
        let mut data = Vec::with_capacity_in(val.data.len(), self.allocator.clone());
        data.extend(val.data.iter().map(|(c, m)| (self.base_ring().clone_el(c), self.clone_monomial(m))));
        MultivariatePolyRingEl {
            data: data
        }
    }

    fn add_ref(&self, lhs: &Self::Element, rhs: &Self::Element) -> Self::Element {
        debug_assert!(self.is_valid(&rhs.data));
        self.add_terms(lhs, rhs.data.iter().map(|(c, m)| (self.base_ring().clone_el(c), m.data.clone().wrap())), Vec::new_in(self.allocator.clone()))
    }

    fn add_assign_ref(&self, lhs: &mut Self::Element, rhs: &Self::Element) {
        *lhs = self.add_ref(lhs, rhs);
    }

    fn add_assign(&self, lhs: &mut Self::Element, rhs: Self::Element) {
        debug_assert!(self.is_valid(&rhs.data));
        *lhs = self.add_terms(&lhs, rhs.data.into_iter(), Vec::new_in(self.allocator.clone()));
    }

    fn sub_assign_ref(&self, lhs: &mut Self::Element, rhs: &Self::Element) {
        debug_assert!(self.is_valid(&rhs.data));
        *lhs = self.add_terms(&lhs, rhs.data.iter().map(|(c, m)| (self.base_ring().negate(self.base_ring.clone_el(c)), m.data.clone().wrap())), Vec::new_in(self.allocator.clone()));
    }
    
    fn negate_inplace(&self, lhs: &mut Self::Element) {
        debug_assert!(self.is_valid(&lhs.data));
        for (c, _) in &mut lhs.data {
            self.base_ring().negate_inplace(c);
        }
    }

    fn mul_assign(&self, lhs: &mut Self::Element, rhs: Self::Element) {
        self.mul_assign_ref(lhs, &rhs);
    }

    fn mul_assign_ref(&self, lhs: &mut Self::Element, rhs: &Self::Element) {
        *lhs = self.mul_ref(lhs, rhs);
    }

    fn mul_ref(&self, lhs: &Self::Element, rhs: &Self::Element) -> Self::Element {
        let mut tmp = Vec::new_in(self.allocator.clone());
        if lhs.data.len() > rhs.data.len() {
            rhs.data.iter().fold(self.zero(), |mut current, (r_c, r_m)| {
                let mut new = self.add_terms(&current, lhs.data.iter().map(|(c, m)| (self.base_ring().mul_ref(c, r_c), self.monomial_mul(m.data.clone().wrap(), r_m))), std::mem::replace(&mut tmp, Vec::new_in(self.allocator.clone())));
                std::mem::swap(&mut new, &mut current);
                std::mem::swap(&mut new.data, &mut tmp);
                current
            })
        } else {
            // we duplicate it to work better with noncommutative rings (not that this is currently of relevance...)
            lhs.data.iter().fold(self.zero(), |mut current, (l_c, l_m)| {
                let mut new = self.add_terms(&current, rhs.data.iter().map(|(c, m)| (self.base_ring().mul_ref(l_c, c), self.monomial_mul(m.data.clone().wrap(), l_m))), std::mem::replace(&mut tmp, Vec::new_in(self.allocator.clone())));
                std::mem::swap(&mut new, &mut current);
                std::mem::swap(&mut new.data, &mut tmp);
                current
            })
        }
    }

    fn zero(&self) -> Self::Element {
        MultivariatePolyRingEl {
            data: Vec::new_in(self.allocator.clone())
        }
    }

    fn from_int(&self, value: i32) -> Self::Element {
        self.from(self.base_ring().get_ring().from_int(value))
    }

    fn eq_el(&self, lhs: &Self::Element, rhs: &Self::Element) -> bool {
        debug_assert!(self.is_valid(&lhs.data));
        debug_assert!(self.is_valid(&rhs.data));
        if lhs.data.len() != rhs.data.len() {
            return false;
        }
        for i in 0..lhs.data.len() {
            if lhs.data.at(i).1.data != rhs.data.at(i).1.data || !self.base_ring.eq_el(&lhs.data.at(i).0, &rhs.data.at(i).0) {
                return false;
            }
        }
        return true;
    }

    fn is_zero(&self, value: &Self::Element) -> bool {
        value.data.len() == 0
    }

    fn is_one(&self, value: &Self::Element) -> bool {
        debug_assert!(self.is_valid(&value.data));
        if value.data.len() != 1 {
            return false;
        }
        value.data[0].1.data.deg() == 0 && self.base_ring().is_one(&value.data[0].0)
    }

    fn is_neg_one(&self, value: &Self::Element) -> bool {
        debug_assert!(self.is_valid(&value.data));
        if value.data.len() != 1 {
            return false;
        }
        value.data[0].1.data.deg() == 0 && self.base_ring().is_neg_one(&value.data[0].0)
    }

    fn is_commutative(&self) -> bool { self.base_ring().is_commutative() }
    fn is_noetherian(&self) -> bool { self.base_ring().is_noetherian() }

    fn dbg(&self, value: &Self::Element, out: &mut std::fmt::Formatter) -> std::fmt::Result {
        self.dbg_within(value, out, EnvBindingStrength::Weakest)
    }

    fn dbg_within<'a>(&self, value: &Self::Element, out: &mut std::fmt::Formatter<'a>, env: EnvBindingStrength) -> std::fmt::Result {
        super::generic_impls::print(RingRef::new(self), value, out, env)
    }

    fn characteristic<I: IntegerRingStore + Copy>(&self, ZZ: I) -> Option<El<I>>
         where I::Type: IntegerRing 
    {
        self.base_ring().characteristic(ZZ)
    }

    fn is_approximate(&self) -> bool {
        self.base_ring().get_ring().is_approximate()
    }
}

/// 
/// Iterator over the terms of a [`MultivariatePolyRingEl`].
/// 
pub struct TermIterImpl<'a, R>
    where R: RingStore
{
    base_iter: std::slice::Iter<'a, (El<R>, MonomialIdentifier)>
}

impl<'a, R> Iterator for TermIterImpl<'a, R>
    where R: RingStore
{
    type Item = (&'a El<R>, &'a MonomialIdentifier);

    fn next(&mut self) -> Option<Self::Item> {
        self.base_iter.next().map(|(c, m)| (c, m))
    }
}

impl<R, A> RingExtension for MultivariatePolyRingImplBase<R, A>
    where R: RingStore,
        A: Clone + Allocator + Send
{
    type BaseRing = R;

    fn base_ring<'b>(&'b self) -> &'b Self::BaseRing {
        &self.base_ring
    }

    fn from(&self, x: El<Self::BaseRing>) -> Self::Element {
        if !self.base_ring().get_ring().is_approximate() && self.base_ring().is_zero(&x) {
            return self.zero();
        } else {
            let mut data = Vec::with_capacity_in(1, self.allocator.clone());
            data.push((x, self.create_monomial((0..self.variable_count).map(|_| 0))));
            return MultivariatePolyRingEl { data };
        }
    }

    fn mul_assign_base(&self, lhs: &mut Self::Element, rhs: &El<Self::BaseRing>) {
        for (c, _) in &mut lhs.data {
            self.base_ring.mul_assign_ref(c, rhs);
        }
        lhs.data.retain(|(c, _)| !self.base_ring.is_zero(c));
    }
}

impl<R, A> MultivariatePolyRing for MultivariatePolyRingImplBase<R, A>
    where R: RingStore,
        A: Clone + Allocator + Send
{
    type Monomial = MonomialIdentifier;
    type TermIter<'a> = TermIterImpl<'a, R>
        where Self: 'a;
        
    fn indeterminate_count(&self) -> usize {
        self.variable_count
    }

    fn create_monomial<I>(&self, exponents: I) -> Self::Monomial
        where I: IntoIterator<Item = usize>,
            I::IntoIter: ExactSizeIterator
    {
        let exponents = exponents.into_iter();
        assert_eq!(exponents.len(), self.indeterminate_count());

        let mut tmp_monomial = self.tmp_monomial2();
        let mut deg = 0;
        for (i, e) in exponents.enumerate() {
            deg += e as Exponent;
            tmp_monomial[i] = e as Exponent;
        }
        assert!(deg <= self.max_supported_deg, "Polynomial ring was configured to support monomials up to degree {}, but create_monomial() was called for degree {}", self.max_supported_deg, deg);

        if deg <= self.safe_degree_threshold {
            // Dense mode: use indexed representation
            return MonomialIdentifier {
                data: InternalMonomialIdentifier {
                    data: InternalMonomialData::Dense {
                        deg: deg,
                        order: enumeration_index_degrevlex(deg, (&*tmp_monomial).clone_els_by(|x| *x), &self.cum_binomial_lookup_table)
                    }
                }
            };
        } else {
            // Sparse mode: store exponents directly
            let exponents_vec: Vec<Exponent> = tmp_monomial.iter().cloned().collect();
            return MonomialIdentifier {
                data: InternalMonomialIdentifier {
                    data: InternalMonomialData::Sparse {
                        deg: deg,
                        exponents: exponents_vec.into_boxed_slice()
                    }
                }
            };
        }
    }

    fn clone_monomial(&self, mon: &Self::Monomial) -> Self::Monomial {
        mon.data.clone().wrap()
    }

    fn add_assign_from_terms<I>(&self, lhs: &mut Self::Element, terms: I)
        where I: IntoIterator<Item = (El<Self::BaseRing>, Self::Monomial)>
    {
        let terms = terms.into_iter();
        let mut rhs = self.tmp_poly.swap(None, std::sync::atomic::Ordering::AcqRel).map(|b| *b).unwrap_or(Vec::new());
        debug_assert!(rhs.len() == 0);
        rhs.extend(terms.into_iter());
        rhs.sort_unstable_by(|l, r| self.compare_degrevlex(&l.1.data, &r.1.data));
        rhs.dedup_by(|(snd_c, snd_m), (fst_c, fst_m)| {
            if self.compare_degrevlex(&fst_m.data, &snd_m.data) == Ordering::Equal {
                self.base_ring().add_assign_ref(fst_c, snd_c);
                return true;
            } else {
                return false;
            }
        });
        *lhs = self.add_terms(&lhs, rhs.drain(..), Vec::new_in(self.allocator.clone()));
        _ = self.tmp_poly.swap(Some(Box::new(rhs)), std::sync::atomic::Ordering::AcqRel);
    }

    fn mul_assign_monomial(&self, f: &mut Self::Element, rhs: Self::Monomial) {
        // Simply use monomial_mul for each term - it already handles sparse/dense properly
        for (_, lhs) in &mut f.data {
            *lhs = self.monomial_mul(lhs.data.clone().wrap(), &rhs);
        }
    }

    fn coefficient_at<'a>(&'a self, f: &'a Self::Element, m: &Self::Monomial) -> &'a El<Self::BaseRing> {
        match f.data.binary_search_by(|(_, fm)| self.compare_degrevlex(&fm.data, &m.data)) {
            Ok(i) => &f.data.at(i).0,
            Err(_) => &self.zero
        }
    }

    fn expand_monomial_to(&self, m: &Self::Monomial, out: &mut [usize]) {
        match &m.data.data {
            InternalMonomialData::Dense { deg, order } => {
                nth_monomial_degrevlex(self.variable_count, *deg, *order, &self.cum_binomial_lookup_table, |i, x| out[i] = x as usize);
            }
            InternalMonomialData::Sparse { exponents, .. } => {
                for (i, &e) in exponents.iter().enumerate() {
                    out[i] = e as usize;
                }
            }
        }
    }

    fn exponent_at(&self, m: &Self::Monomial, var_index: usize) -> usize {
        match &m.data.data {
            InternalMonomialData::Dense { deg, order } => {
                let mut output = 0;
                nth_monomial_degrevlex(self.variable_count, *deg, *order, &self.cum_binomial_lookup_table, |i, x| if i == var_index { output = x });
                output as usize
            }
            InternalMonomialData::Sparse { exponents, .. } => {
                exponents[var_index] as usize
            }
        }
    }

    fn terms<'a>(&'a self, f: &'a Self::Element) -> Self::TermIter<'a> {
        TermIterImpl {
            base_iter: f.data.iter()
        }
    }

    fn monomial_deg(&self, mon: &Self::Monomial) -> usize {
        mon.data.deg() as usize
    }

    fn LT<'a, O: MonomialOrder>(&'a self, f: &'a Self::Element, order: O) -> Option<(&'a El<Self::BaseRing>, &'a Self::Monomial)> {
        if order.is_same(&DegRevLex) {
            f.data.last().map(|(c, m)| (c, m))
        } else {
            self.terms(f).max_by(|l, r| order.compare(RingRef::new(self), &l.1, &r.1))
        }
    }

    fn largest_term_lt<'a, O: MonomialOrder>(&'a self, f: &'a Self::Element, order: O, lt_than: &Self::Monomial) -> Option<(&'a El<Self::BaseRing>, &'a Self::Monomial)> {
        if order.is_same(&DegRevLex) {
            let res = match f.data.binary_search_by(|(_, mon)| self.compare_degrevlex(&mon.data, &lt_than.data)) {
                Ok(0) => None,
                Ok(i) => Some((&f.data[i - 1].0, &f.data[i - 1].1)),
                Err(0) => None,
                Err(i) => Some((&f.data[i - 1].0, &f.data[i - 1].1))
            };
            assert!({
                let expected = self.terms(f).filter(|(_, m)| order.compare(RingRef::new(self), m, lt_than) == Ordering::Less).max_by(|l, r| order.compare(RingRef::new(self), &l.1, &r.1));
                (res.is_none() && expected.is_none()) || std::ptr::eq(res.unwrap().0, expected.unwrap().0)
            });
            return res;
        } else {
            self.terms(f).filter(|(_, m)| order.compare(RingRef::new(self), m, lt_than) == Ordering::Less).max_by(|l, r| order.compare(RingRef::new(self), &l.1, &r.1))
        }
    }

    fn monomial_mul(&self, lhs: Self::Monomial, rhs: &Self::Monomial) -> Self::Monomial {
        let lhs_deg = lhs.data.deg();
        let rhs_deg = rhs.data.deg();

        // Try to use multiplication table (only works for dense mode)
        if lhs_deg <= rhs_deg {
            if let (InternalMonomialData::Dense { order: lhs_order, .. }, InternalMonomialData::Dense { order: rhs_order, .. }) = (&lhs.data.data, &rhs.data.data) {
                if let Some(table) = self.try_get_multiplication_table(lhs_deg, rhs_deg) {
                    return MonomialIdentifier {
                        data: InternalMonomialIdentifier {
                            data: InternalMonomialData::Dense {
                                deg: lhs_deg + rhs_deg,
                                order: table[*lhs_order as usize][*rhs_order as usize]
                            }
                        }
                    };
                }
            }
        } else {
            if let (InternalMonomialData::Dense { order: lhs_order, .. }, InternalMonomialData::Dense { order: rhs_order, .. }) = (&lhs.data.data, &rhs.data.data) {
                if let Some(table) = self.try_get_multiplication_table(rhs_deg, lhs_deg) {
                    return MonomialIdentifier {
                        data: InternalMonomialIdentifier {
                            data: InternalMonomialData::Dense {
                                deg: lhs_deg + rhs_deg,
                                order: table[*rhs_order as usize][*lhs_order as usize]
                            }
                        }
                    };
                }
            }
        }

        // Fallback to exponent-wise operation
        return self.exponent_wise_bivariate_monomial_operation(lhs.data, rhs.data.clone(), |a, b| a + b);
    }

    fn monomial_lcm(&self, lhs: Self::Monomial, rhs: &Self::Monomial) -> Self::Monomial {
        self.exponent_wise_bivariate_monomial_operation(lhs.data, rhs.data.clone(), |a, b| max(a, b))
    }

    fn monomial_div(&self, lhs: Self::Monomial, rhs: &Self::Monomial) -> Result<Self::Monomial, Self::Monomial> {
        let mut failed = false;
        let result = self.exponent_wise_bivariate_monomial_operation(lhs.data, rhs.data.clone(), |a, b| match a.checked_sub(b) {
            Some(x) => x,
            None => {
                failed = true;
                0
            }
        });
        if failed {
            Err(result)
        } else {
            Ok(result)
        }
    }

    fn evaluate<S, V, H>(&self, f: &Self::Element, values: V, hom: H) -> S::Element
        where S: ?Sized + RingBase,
            H: Homomorphism<<Self::BaseRing as RingStore>::Type, S>,
            V: VectorFn<S::Element>
    {
        assert!(hom.domain().get_ring() == self.base_ring().get_ring());
        assert_eq!(values.len(), self.indeterminate_count());
        let new_ring = RingValue::from(MultivariatePolyRingImplBase {
            zero: hom.codomain().zero(),
            base_ring: hom.codomain(),
            variable_count: self.variable_count,
            max_supported_deg: self.max_supported_deg,
            safe_degree_threshold: self.safe_degree_threshold,
            monomial_multiplication_table: self.monomial_multiplication_table.clone(),
            tmp_monomials: ThreadLocal::new(),
            cum_binomial_lookup_table: self.cum_binomial_lookup_table.clone(),
            tmp_poly: AtomicOptionBox::new(None),
            allocator: self.allocator.clone()
        });
        let mut result = new_ring.from_terms(self.terms(f).map(|(c, m)| (hom.map_ref(c), new_ring.create_monomial((0..self.indeterminate_count()).map(|i| self.exponent_at(m, i))))));
        for i in 0..self.indeterminate_count() {
            result = new_ring.specialize(&result, i, &new_ring.inclusion().map(values.at(i)));
        }
        debug_assert!(result.data.len() <= 1);
        if result.data.len() == 0 {
            return hom.codomain().zero();
        } else {
            debug_assert!(result.data[0].1.data.deg() == 0);
            return result.data.into_iter().next().unwrap().0;
        }
    }
}

impl<P, R, A> CanHomFrom<P> for MultivariatePolyRingImplBase<R, A> 
    where R: RingStore,
        A: Clone + Allocator + Send,
        P: MultivariatePolyRing,
        R::Type: CanHomFrom<<P::BaseRing as RingStore>::Type>
{
    type Homomorphism = <R::Type as CanHomFrom<<P::BaseRing as RingStore>::Type>>::Homomorphism;

    fn has_canonical_hom(&self, from: &P) -> Option<Self::Homomorphism> {
        if self.indeterminate_count() >= from.indeterminate_count() {
            self.base_ring().get_ring().has_canonical_hom(from.base_ring().get_ring())
        } else {
            None
        }
    }

    fn map_in(&self, from: &P, el: <P as RingBase>::Element, hom: &Self::Homomorphism) -> Self::Element {
        self.map_in_ref(from, &el, hom)
    }

    fn map_in_ref(&self, from: &P, el: &<P as RingBase>::Element, hom: &Self::Homomorphism) -> Self::Element {
        RingRef::new(self).from_terms(from.terms(el).map(|(c, m)| (
            self.base_ring().get_ring().map_in_ref(from.base_ring().get_ring(), c, hom),
            self.create_monomial((0..self.indeterminate_count()).map(|i| if i < from.indeterminate_count() { from.exponent_at(m, i) } else { 0 }))
        )))
    }
}

impl<P, R, A> CanIsoFromTo<P> for MultivariatePolyRingImplBase<R, A> 
    where R: RingStore,
        A: Clone + Allocator + Send,
        P: MultivariatePolyRing,
        R::Type: CanIsoFromTo<<P::BaseRing as RingStore>::Type>
{
    type Isomorphism = <R::Type as CanIsoFromTo<<P::BaseRing as RingStore>::Type>>::Isomorphism;

    fn has_canonical_iso(&self, from: &P) -> Option<Self::Isomorphism> {
        if self.indeterminate_count() == from.indeterminate_count() {
            self.base_ring().get_ring().has_canonical_iso(from.base_ring().get_ring())
        } else {
            None
        }
    }

    fn map_out(&self, from: &P, el: Self::Element, iso: &Self::Isomorphism) -> <P as RingBase>::Element {
        RingRef::new(from).from_terms(self.terms(&el).map(|(c, m)| (
            self.base_ring().get_ring().map_out(from.base_ring().get_ring(), self.base_ring().clone_el(c), iso),
            from.create_monomial((0..self.indeterminate_count()).map(|i| self.exponent_at(m, i)))
        )))
    }
}

#[cfg(test)]
use crate::rings::zn::zn_static;
#[cfg(test)]
use crate::rings::zn::zn_static::F17;
#[cfg(test)]
use crate::rings::zn::*;
#[cfg(test)]
use crate::rings::approx_real::float::Real64;

#[cfg(test)]
fn ring_and_elements() -> (MultivariatePolyRingImpl<zn_static::Fp<17>>, Vec<MultivariatePolyRingEl<zn_static::Fp<17>>>) {
    let ring = MultivariatePolyRingImpl::new(F17, 3);
    let els = vec![
        ring.from_terms([]),
        ring.from_terms([(ring.base_ring().one(), ring.create_monomial([0, 0, 0]))]),
        ring.from_terms([(ring.base_ring().neg_one(), ring.create_monomial([0, 0, 0]))]),
        ring.from_terms([(ring.base_ring().one(), ring.create_monomial([1, 0, 0]))]),
        ring.from_terms([(ring.base_ring().neg_one(), ring.create_monomial([1, 0, 0]))]),
        ring.from_terms([(ring.base_ring().one(), ring.create_monomial([1, 0, 0])), (ring.base_ring().one(), ring.create_monomial([0, 1, 0]))]),
        ring.from_terms([(ring.base_ring().one(), ring.create_monomial([2, 0, 0])), (ring.base_ring().neg_one(), ring.create_monomial([1, 0, 0]))]),
        ring.from_terms([(ring.base_ring().one(), ring.create_monomial([1, 0, 0])), (ring.base_ring().neg_one(), ring.create_monomial([0, 1, 1])), (ring.base_ring().one(), ring.create_monomial([0, 0, 2]))]),
    ];
    return (ring, els);
}

#[test]
fn test_ring_axioms() {
    let (ring, els) = ring_and_elements();
    crate::ring::generic_tests::test_ring_axioms(&ring, els.into_iter());
}

#[test]
fn test_multivariate_axioms() {
    let (ring, _els) = ring_and_elements();
    crate::rings::multivariate::generic_tests::test_poly_ring_axioms(&ring, [F17.one(), F17.zero(), F17.int_hom().map(2), F17.neg_one()].into_iter());
}

#[test]
fn test_enumeration_index_degrevlex() {

    let cum_binomial_lookup_table = (0..4).map(|n| (0..7).map(|k| compute_cum_binomial(n, k)).collect::<Vec<_>>()).collect::<Vec<_>>();

    assert_eq!(0, enumeration_index_degrevlex(0, [0, 0, 0, 0].clone_els_by(|x| *x), &cum_binomial_lookup_table));
    let mut check = [0, 0, 0, 0];
    nth_monomial_degrevlex(4, 0, 0, &cum_binomial_lookup_table, |i, x| check[i] = x);
    assert_eq!(&[0, 0, 0, 0], &check);

    assert_eq!(0, enumeration_index_degrevlex(1, [0, 0, 0, 1].clone_els_by(|x| *x), &cum_binomial_lookup_table));
    let mut check = [0, 0, 0, 0];
    nth_monomial_degrevlex(4, 1, 0, &cum_binomial_lookup_table, |i, x| check[i] = x);
    assert_eq!(&[0, 0, 0, 1], &check);

    assert_eq!(3, enumeration_index_degrevlex(1, [1, 0, 0, 0].clone_els_by(|x| *x), &cum_binomial_lookup_table));
    let mut check = [0, 0, 0, 0];
    nth_monomial_degrevlex(4, 1, 3, &cum_binomial_lookup_table, |i, x| check[i] = x);
    assert_eq!(&[1, 0, 0, 0], &check);

    fn degrevlex_cmp(lhs: &[u16; 4], rhs: &[u16; 4]) -> Ordering {
        let lhs_deg = lhs[0] + lhs[1] + lhs[2] + lhs[3];
        let rhs_deg = rhs[0] + rhs[1] + rhs[2] + rhs[3];
        if lhs_deg < rhs_deg {
            return Ordering::Less;
        } else if lhs_deg > rhs_deg {
            return Ordering::Greater;
        } else {
            for i in (0..4).rev() {
                if lhs[i] > rhs[i] {
                    return Ordering::Less
                } else if lhs[i] < rhs[i] {
                    return Ordering::Greater;
                }
            }
            return Ordering::Equal;
        }
    }

    let all_monomials = multiset_combinations(&[6, 6, 6, 6], 6, |slice| std::array::from_fn::<_, 4, _>(|i| slice[3 - i] as u16)).collect::<Vec<_>>();
    assert!(all_monomials.is_sorted_by(|l, r| degrevlex_cmp(l, r) == Ordering::Less));

    for i in 0..all_monomials.len() {
        assert_eq!(i as u64, enumeration_index_degrevlex(6, (&all_monomials[i]).clone_els_by(|x| *x), &cum_binomial_lookup_table));
        let mut check = [0, 0, 0, 0];
        nth_monomial_degrevlex(4, 6, i as u64, &cum_binomial_lookup_table, |i, x| check[i] = x);
        assert_eq!(&all_monomials[i], &check);
    }
}

#[test]
fn test_create_multiplication_table() {
    let cum_binomial_lookup_table = (0..3).map(|n| (0..7).map(|k| compute_cum_binomial(n, k)).collect::<Vec<_>>()).collect::<Vec<_>>();
    let mul_table = MultivariatePolyRingImplBase::<StaticRing<i64>>::create_multiplication_table(3, 3, 4, &cum_binomial_lookup_table);

    let deg3_monomials = multiset_combinations(&[3, 3, 3], 3, |slice| std::array::from_fn::<_, 3, _>(|i| slice[2 - i] as u16)).collect::<Vec<_>>();
    let deg4_monomials = multiset_combinations(&[4, 4, 4], 4, |slice| std::array::from_fn::<_, 3, _>(|i| slice[2 - i] as u16)).collect::<Vec<_>>();

    for lhs in &deg3_monomials {
        for rhs in &deg4_monomials {
            assert_eq!(
                enumeration_index_degrevlex(7, (0..3).map_fn(|i| lhs[i] + rhs[i]), &cum_binomial_lookup_table),
                mul_table[
                    enumeration_index_degrevlex(3, (0..3).map_fn(|i| lhs[i]), &cum_binomial_lookup_table) as usize
                ][
                    enumeration_index_degrevlex(4, (0..3).map_fn(|i| rhs[i]), &cum_binomial_lookup_table) as usize
                ]
            );
        }
    }
}

#[test]
fn test_monomial_small() {
    assert_eq!(16, size_of::<MonomialIdentifier>());
}

#[test]
fn test_new_many_variables() {
    for m in 1..32 {
        let ring = MultivariatePolyRingImpl::new_with_mult_table(StaticRing::<i64>::RING, m, 32, (2, 3), Global);
        assert_eq!(m, ring.indeterminate_count());
    }
}

#[test]
fn test_evaluate_approximate_ring() {
    let ring = MultivariatePolyRingImpl::new(Real64::RING, 2);
    let [f] = ring.with_wrapped_indeterminates(|[X, Y]| [X * X * Y - Y * Y]);
    let x = 0.47312;
    let y = -1.43877;
    assert!(Real64::RING.abs((x * x * y - y * y) - ring.evaluate(&f, [x, y].clone_els_by(|x| *x), &Real64::RING.identity())) <= 0.000000001);
}

#[test]
fn test_evaluate_many_variables() {
    let ring = MultivariatePolyRingImpl::new_with_mult_table(StaticRing::<i64>::RING, 20, 16, (4, 6), Global);
    let [f] = ring.with_wrapped_indeterminates(|X: [_; 20]| [X[0] + X[5] + X[19]]);
    assert_eq!(1 + 6 + 20, ring.evaluate(&f, (1..21).map_fn(|x| x as i64), StaticRing::<i64>::RING.identity()));
}

#[test]
fn test_appearing_indeterminates() {
    let F7 = zn_64::Zn::new(7).as_field().ok().unwrap();
    let F7XY = MultivariatePolyRingImpl::new(&F7, 2);
    let [f, g] = F7XY.with_wrapped_indeterminates(|[X, Y]| [5 + 4 * X, 6 + 2 * Y]);
    assert_eq!(vec![(0, 1)], F7XY.appearing_indeterminates(&f));
    assert_eq!(vec![(1, 1)], F7XY.appearing_indeterminates(&g));
}

#[test]
fn test_64_variables_sparse_mode() {
    // This test validates the sparse monomial infrastructure for large-variable systems
    // Context: zyga ZKP requires 64 variables with degrees up to 30+

    let ring = MultivariatePolyRingImpl::new_with_mult_table(
        StaticRing::<i64>::RING,
        64,        // 64 variables
        30,        // max degree 30
        (2, 2),    // minimal multiplication table
        Global
    );

    // Verify safe_degree_threshold was computed
    // For 64 variables, binomial(64+d-1, d) exceeds u64::MAX somewhere around d=21
    // The exact threshold depends on the computation, but it should be > 0
    assert!(ring.get_ring().safe_degree_threshold > 0);
    assert!(ring.get_ring().safe_degree_threshold < ring.get_ring().max_supported_deg);

    // Test 1: Low-degree monomials should use dense mode
    let m_deg1 = ring.create_monomial((0..64).map(|i| if i == 0 { 1 } else { 0 })); // X_0
    let m_deg2 = ring.create_monomial((0..64).map(|i| if i == 0 { 2 } else { 0 })); // X_0^2

    // Verify these use dense mode by checking the internal representation
    match &m_deg1.data.data {
        InternalMonomialData::Dense { deg, .. } => assert_eq!(*deg, 1),
        InternalMonomialData::Sparse { .. } => panic!("Expected dense mode for degree 1"),
    }
    match &m_deg2.data.data {
        InternalMonomialData::Dense { deg, .. } => assert_eq!(*deg, 2),
        InternalMonomialData::Sparse { .. } => panic!("Expected dense mode for degree 2"),
    }

    // Test 2: High-degree monomials should use sparse mode
    // Create a monomial with degree > safe_degree_threshold
    let high_deg = ring.get_ring().safe_degree_threshold + 1;
    let m_high = ring.create_monomial((0..64).map(|i| if i == 0 { high_deg as usize } else { 0 }));

    match &m_high.data.data {
        InternalMonomialData::Sparse { deg, exponents } => {
            assert_eq!(*deg, high_deg);
            assert_eq!(exponents.len(), 64);
            assert_eq!(exponents[0], high_deg);
            for i in 1..64 {
                assert_eq!(exponents[i], 0);
            }
        }
        InternalMonomialData::Dense { .. } => panic!("Expected sparse mode for degree {}", high_deg),
    }

    // Test 3: Monomial multiplication across modes (dense * dense -> sparse)
    if ring.get_ring().safe_degree_threshold >= 2 {
        let m1 = ring.create_monomial((0..64).map(|i| if i == 0 { 1 } else { 0 }));
        let m2 = ring.create_monomial((0..64).map(|i| if i == 1 { 1 } else { 0 }));
        let m_prod = ring.monomial_mul(m1, &m2);

        // Product is X_0 * X_1, degree 2
        let mut expected_exps = vec![0u16; 64];
        expected_exps[0] = 1;
        expected_exps[1] = 1;

        let mut actual_exps = vec![0; 64];
        ring.expand_monomial_to(&m_prod, &mut actual_exps);

        assert_eq!(expected_exps.iter().map(|&x| x as usize).collect::<Vec<_>>(), actual_exps);
    }

    // Test 4: Sparse monomial multiplication (sparse * sparse -> sparse)
    let deg_a = ring.get_ring().safe_degree_threshold + 1;
    let deg_b = 2;
    let m_sparse1 = ring.create_monomial((0..64).map(|i| if i == 5 { deg_a as usize } else { 0 }));
    let m_sparse2 = ring.create_monomial((0..64).map(|i| if i == 10 { deg_b as usize } else { 0 }));
    let m_sparse_prod = ring.monomial_mul(ring.clone_monomial(&m_sparse1), &m_sparse2);

    match &m_sparse_prod.data.data {
        InternalMonomialData::Sparse { deg, exponents } => {
            assert_eq!(*deg, deg_a + deg_b);
            assert_eq!(exponents[5], deg_a);
            assert_eq!(exponents[10], deg_b);
        }
        InternalMonomialData::Dense { .. } => panic!("Expected sparse mode for high-degree product"),
    }

    // Test 5: Monomial LCM with sparse monomials
    let m_lcm = ring.monomial_lcm(m_sparse1, &m_sparse2);
    assert_eq!(ring.monomial_deg(&m_lcm), (deg_a + deg_b) as usize);

    // Test 6: Polynomial arithmetic with sparse monomials
    let high_deg_term = ring.get_ring().safe_degree_threshold + 2;
    let poly_sparse = ring.from_terms([
        (ring.base_ring().one(), ring.create_monomial((0..64).map(|i| if i == 0 { high_deg_term as usize } else { 0 }))),
        (ring.base_ring().neg_one(), ring.create_monomial((0..64).map(|i| if i == 1 { high_deg_term as usize } else { 0 }))),
    ]);

    // Verify polynomial operations work
    let poly_sum = ring.add_ref(&poly_sparse, &poly_sparse);
    assert_eq!(ring.terms(&poly_sum).count(), 2); // Two terms should combine (coeff doubling)

    // Test 7: DegRevLex comparison for sparse monomials
    let m_a = ring.create_monomial((0..64).map(|i| if i == 63 { high_deg_term as usize } else { 0 }));
    let m_b = ring.create_monomial((0..64).map(|i| if i == 0 { high_deg_term as usize } else { 0 }));

    // In DegRevLex, same degree is compared by reverse lex: [0,0,...,d] < [d,0,...,0]
    let cmp = ring.get_ring().compare_degrevlex(&m_a.data, &m_b.data);
    assert_eq!(cmp, Ordering::Less, "DegRevLex comparison failed for sparse monomials");

    // Test 8: Exponent access for sparse monomials
    assert_eq!(ring.exponent_at(&m_high, 0), high_deg as usize);
    assert_eq!(ring.exponent_at(&m_high, 1), 0);
    assert_eq!(ring.exponent_at(&m_high, 63), 0);
}

#[test]
fn test_sparse_mode_polynomial_multiplication() {
    // Test polynomial multiplication with sparse monomials
    let ring = MultivariatePolyRingImpl::new_with_mult_table(
        StaticRing::<i64>::RING,
        64,
        30,
        (1, 1),
        Global
    );

    let high_deg = ring.get_ring().safe_degree_threshold + 1;

    // Create two polynomials with sparse monomials
    // f = X_0^high_deg + X_1^high_deg
    let f = ring.from_terms([
        (1, ring.create_monomial((0..64).map(|i| if i == 0 { high_deg as usize } else { 0 }))),
        (1, ring.create_monomial((0..64).map(|i| if i == 1 { high_deg as usize } else { 0 }))),
    ]);

    // g = X_2 + 1
    let g = ring.from_terms([
        (1, ring.create_monomial((0..64).map(|i| if i == 2 { 1 } else { 0 }))),
        (1, ring.create_monomial((0..64).map(|_| 0))),
    ]);

    // Compute f * g
    let product = ring.mul_ref(&f, &g);

    // Expected: X_0^high_deg * X_2 + X_0^high_deg + X_1^high_deg * X_2 + X_1^high_deg
    assert_eq!(ring.terms(&product).count(), 4);

    // Verify degrees
    for (_, m) in ring.terms(&product) {
        let deg = ring.monomial_deg(m);
        assert!(deg == high_deg as usize || deg == (high_deg + 1) as usize);
    }
}