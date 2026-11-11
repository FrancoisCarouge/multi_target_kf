// The adapter of this package between the Kalman, TypedLinearAlgebra, and SparseLinearAlgebra libraries and
// mp-units: what the typed motion models need, and nothing else.
#ifndef TYPED_LINEAR_ALGEBRA_H
#define TYPED_LINEAR_ALGEBRA_H

#include <concepts>
#include <cstddef>
#include <tuple>
#include <type_traits>

#include "fcarouge/kalman.hpp"
#include "fcarouge/sparse_linear_algebra.hpp"
#include "fcarouge/typed_linear_algebra.hpp"
#include "mp-units/framework/quantity.h"
#include "mp-units/math.h"
#include "mp-units/systems/si.h"

namespace fcarouge
{
// A quantity element is stored as its numerical value in its own unit, and a stored value is read back as a
// quantity of that unit.
template <typename To, mp_units::Quantity From>
struct element_caster<To, From>
{
  [[nodiscard]] static constexpr auto operator()(From value) -> To
  {
    static_assert(std::same_as<typename From::rep, std::remove_cvref_t<To>>);
    return value.numerical_value_in(From::unit);
  }
};

template <mp_units::Quantity To, typename From>
struct element_caster<To, From>
{
  [[nodiscard]] static constexpr auto operator()(From value) -> To
  {
    static_assert(std::same_as<typename To::rep, std::remove_cvref_t<From>>);
    return value * To::reference;
  }
};

// A quantity reference cannot be materialized out of a stored number: a reference read gives a constant
// value, so that an assignment through it, a write that would be lost, does not compile.
template <mp_units::Quantity To, typename From>
struct element_caster<To &, From &>
{
  [[nodiscard]] static constexpr auto operator()(From value) -> const To
  {
    static_assert(std::same_as<typename To::rep, std::remove_cvref_t<From>>);
    return value * To::reference;
  }
};

// Kalman names the matrices of a filter by dividing its column vectors, F = state / state, H = output / state,
// a division SparseLinearAlgebra rejects for a non-square denominator. Declared only: it names the type, and a
// run-time use fails to link.
template <typename Type, auto Lhs, auto Rhs>
  requires(Lhs.columns == 1 && Rhs.columns == 1 && Rhs.rows > 1)
auto operator/(const sparse_matrix<Type, Lhs> &, const sparse_matrix<Type, Rhs> &)
    -> sparse_matrix<Type, sparse::dense<Lhs.rows, Rhs.rows>>;
}  // namespace fcarouge

namespace fcarouge::kalman_filter::internal
{
// The transpose of a SparseLinearAlgebra storage, through its free transposed().
template <typename Type, auto Pattern>
struct transposes<sparse_matrix<Type, Pattern>>
{
  [[nodiscard]] static constexpr auto operator()(const sparse_matrix<Type, Pattern> &value)
  {
    return transposed(value);
  }
};

// The transpose of a typed matrix: the transposed storage, the indexes exchanged.
template <typename Matrix, typename RowIndexes, typename ColumnIndexes>
struct transposes<typed_matrix<Matrix, RowIndexes, ColumnIndexes>>
{
  [[nodiscard]] static constexpr auto operator()(
    const typed_matrix<Matrix, RowIndexes, ColumnIndexes> &value)
  {
    return typed_matrix<evaluate<transpose<Matrix>>, ColumnIndexes, RowIndexes>{t(value.data())};
  }
};

// The zero matrix of a SparseLinearAlgebra storage: zero initialized.
template <typename Type, auto Pattern>
inline constexpr sparse_matrix<Type, Pattern> zero<sparse_matrix<Type, Pattern>>{};

// The one matrix of a SparseLinearAlgebra storage: ones on its stored diagonal.
template <typename Type, auto Pattern>
inline constexpr sparse_matrix<Type, Pattern> one<sparse_matrix<Type, Pattern>>{[] {
  sparse_matrix<Type, Pattern> value;
  for (std::size_t i{0}; i < Pattern.rows && i < Pattern.columns; ++i) {
    if (Pattern.stores(i, i)) {
      value.values[Pattern.index_of(i, i)] = Type{1};
    }
  }
  return value;
}()};

// The zero and one typed matrices, of the zero and one storages.
template <typename Matrix, typename RowIndexes, typename ColumnIndexes>
inline constexpr typed_matrix<Matrix, RowIndexes, ColumnIndexes>
  zero<typed_matrix<Matrix, RowIndexes, ColumnIndexes>>{zero<Matrix>};

template <typename Matrix, typename RowIndexes, typename ColumnIndexes>
inline constexpr typed_matrix<Matrix, RowIndexes, ColumnIndexes>
  one<typed_matrix<Matrix, RowIndexes, ColumnIndexes>>{one<Matrix>};
}  // namespace fcarouge::kalman_filter::internal

namespace multi_target_kf::typed
{
template <auto Reference>
using Quantity = mp_units::quantity<Reference, double>;

using Length = Quantity<mp_units::si::metre>;
using Speed = Quantity<mp_units::si::metre / mp_units::si::second>;
using Acceleration = Quantity<mp_units::si::metre / (mp_units::si::second * mp_units::si::second)>;
using Time = Quantity<mp_units::si::second>;

/// A column vector of the given element types.
template <typename... Types>
using Vector = fcarouge::typed_column_vector<
  fcarouge::sparse_matrix<double, fcarouge::sparse::dense<sizeof...(Types), 1>>, Types...>;

/// A dense matrix indexed by the rows of a column vector and the columns of another.
template <typename Rows, typename Columns>
using Matrix = fcarouge::typed_matrix<
  fcarouge::sparse_matrix<double, fcarouge::sparse::dense<Rows::rows, Columns::rows>>,
  typename Rows::row_indexes, typename Columns::row_indexes>;

/// The type of an element of a typed matrix.
template <typename Typed, std::size_t Row, std::size_t Column = 0>
using Element = fcarouge::tla::element_at<Typed, Row, Column>;

/// The numerical value of an element, in the unit of its type.
template <std::size_t Row, std::size_t Column, typename Typed>
[[nodiscard]] constexpr double value_of(const Typed & matrix)
{
  return matrix.template at<Row, Column>().numerical_value_in(Element<Typed, Row, Column>::unit);
}

template <std::size_t Row, typename Typed>
[[nodiscard]] constexpr double value_of(const Typed & vector)
{
  return vector.template at<Row>().numerical_value_in(Element<Typed, Row>::unit);
}
}  // namespace multi_target_kf::typed

#endif  // TYPED_LINEAR_ALGEBRA_H
