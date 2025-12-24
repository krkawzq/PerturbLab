// sparse.hpp - Minimal Sparse Matrix Views (no torch, no eigen)
#pragma once
#include "macro.hpp"
#include <cstddef>
#include <cstdint>

namespace perturblab {
namespace kernel {
namespace view {

/**
 * @brief CSR (Compressed Sparse Row) matrix view.
 * 
 * Representation:
 * - data: Non-zero values.
 * - indices: Column indices of non-zero values.
 * - indptr: Pointers to the start of each row in data and indices.
 * 
 * @tparam T Numeric type.
 */
template<class T>
struct CsrView {
    const T*       data_;     // Non-zero values (nnz)
    const int64_t* indices_;  // Column indices (nnz)
    const int64_t* indptr_;   // Row pointers (rows + 1)
    size_t rows_;
    size_t cols_;
    size_t nnz_;

    force_inline_ const T*       data()    const { return data_; }
    force_inline_ const int64_t* indices() const { return indices_; }
    force_inline_ const int64_t* indptr()  const { return indptr_; }
    force_inline_ size_t rows()  const { return rows_; }
    force_inline_ size_t cols()  const { return cols_; }
    force_inline_ size_t nnz()   const { return nnz_; }

    static force_inline_ CsrView from_raw(const T* data,
                            const int64_t* indices,
                            const int64_t* indptr,
                            size_t rows, size_t cols, size_t nnz) {
        return CsrView(data, indices, indptr, rows, cols, nnz);
    }

private:
    CsrView(const T* d, const int64_t* i, const int64_t* p,
            size_t r, size_t c, size_t n)
        : data_(d), indices_(i), indptr_(p), rows_(r), cols_(c), nnz_(n) {}
};

/**
 * @brief CSC (Compressed Sparse Column) matrix view.
 * 
 * Representation:
 * - data: Non-zero values.
 * - indices: Row indices of non-zero values.
 * - indptr: Pointers to the start of each column in data and indices.
 * 
 * @tparam T Numeric type.
 */
template<class T>
struct CscView {
    const T*       data_;     // Non-zero values (nnz)
    const int64_t* indices_;  // Row indices (nnz)
    const int64_t* indptr_;   // Column pointers (cols + 1)
    size_t rows_;
    size_t cols_;
    size_t nnz_;

    force_inline_ const T*       data()    const { return data_; }
    force_inline_ const int64_t* indices() const { return indices_; }
    force_inline_ const int64_t* indptr()  const { return indptr_; }
    force_inline_ size_t rows()  const { return rows_; }
    force_inline_ size_t cols()  const { return cols_; }
    force_inline_ size_t nnz()   const { return nnz_; }

    static force_inline_ CscView from_raw(const T* data,
                            const int64_t* indices,
                            const int64_t* indptr,
                            size_t rows, size_t cols, size_t nnz) {
        return CscView(data, indices, indptr, rows, cols, nnz);
    }

private:
    CscView(const T* d, const int64_t* i, const int64_t* p,
            size_t r, size_t c, size_t n)
        : data_(d), indices_(i), indptr_(p), rows_(r), cols_(c), nnz_(n) {}
};

} // namespace view
} // namespace kernel
} // namespace perturblab
